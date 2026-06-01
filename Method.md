# Method: Two Workflows for Object Insertion

## Overview

OpenSplice provides **two complementary workflows** in a tabbed Gradio UI:

**Tab 1 — Interactive Placement**: Free-form drag, rotate, and scale the foreground anywhere on the background. Best for creative placement where the user wants full control.

**Tab 2 — SAM3 Stitch**: Segment a specific object in the background, then replace it with another. Best for targeted object replacement (e.g., swap a face, replace a product).

Both share the same core blending and harmonization modules.

---

## Tab 1: Interactive Placement

```
Background Upload           Foreground Upload / AI Generate
        │                              │
        │                     SAM3 center-point auto-segment
        │                         → largest mask = object
        │                              │
        └──────────┬───────────────────┘
                   │
            Live Preview
        (click to set position,
         sliders for rotation & scale)
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
Alpha Blend   Poisson Blend   AI Harmonize
(cut-paste)   (seamlessClone)  (DashScope)
                   │
          ┌────────┴────────┐
          ▼                 ▼
    Reinhard Color      simOPA Score
    Transfer            (0–1 rating)
```

### Auto-Segmentation of Foreground

When a foreground image is uploaded or generated:

1. Create a `Segmenter` instance for the foreground
2. Call `segment_by_point([[cx, cy]], [1])` with the image center — `multimask=True` gives 3 candidates
3. Select the mask with the largest area (`mask.sum()`)
4. Use this mask to isolate the foreground object from its background

### Transform Pipeline — `transforms.py`

**Rotation + Scaling** (`apply_transform`):
- Use `cv2.getRotationMatrix2D` around the image center
- Auto-expand the output canvas (accounting for `cos`/`sin` of rotation angle) so no clipping occurs
- Apply `cv2.warpAffine` with `INTER_CUBIC` for the image, `INTER_LINEAR` for the mask
- Mask is scaled ×255 before interpolation, thresholded at 128 after (handles binary mask correctly)

**Live Preview** (`render_overlay`):
- Transform the foreground + mask
- Alpha-composite onto the background at the user-chosen center position
- Clip to background bounds (partial off-screen rendering supported)

### Alpha Blend — `alpha_place()`

The simplest form of compositing:

```
result = foreground × mask + background × (1 − mask)
```

Direct pixel replacement where mask > 0. Fastest option. No gradient blending — sharp edges between foreground and background.

### Poisson Blend — `place_and_blend()`

Calls `cv2.seamlessClone` with `NORMAL_CLONE`:

1. Transform foreground and mask (rotation + scale)
2. Place onto a full-size canvas at the chosen position
3. Gaussian-blur the mask edges (kernel size 5)
4. Threshold at 128 (mask is 0-255 after ×255 step)
5. `cv2.seamlessClone(fg_full, background, mask_uint8, center, NORMAL_CLONE)`

`NORMAL_CLONE` preserves the foreground texture while adapting gradients at the mask boundary to match the background. Falls back to alpha compositing on error.

---

## Tab 2: SAM3 Stitch

```
Upload Image
      │
SAM 3 Segmentation (text / box / point)
      │
Select Mask → Upload or Generate Replacement
      │
crop_to_object() → remove background from replacement
      │
resize_and_crop_to_mask() → match aspect ratio
      │
poisson_blend() → seamlessClone at mask bbox
      │
      ├── Fast Stitch (done)
      └── AI Harmonize (DashScope fix-up)
```

### Segmentation — SAM 3

Three prompt modes:

| Mode | API | Note |
|------|-----|------|
| **Text** | `Sam3Processor.set_text_prompt()` → ground → `predict_inst()` per box | Open-vocabulary, Chinese + English |
| **Box** | `model.predict_inst(box=[x1,y1,x2,y2])` | Click two corners |
| **Point** | `model.predict_inst(point_coords=..., point_labels=...)` | Click foreground points |

Text grounding: detects all instances matching a concept, returns coarse boxes. Each box is refined through `predict_inst` for a high-quality mask.

### Object Extraction — `crop_to_object()`

AI-generated images include background. The algorithm (pure OpenCV):

1. Canny edge detection (30/100)
2. Morphological close (7×7 elliptical kernel, 2 iterations)
3. Find external contours, filter near image center (within 40% of longest dimension)
4. Combined bounding box + 8px padding → crop

Fast (~milliseconds), no extra inference cost.

### Poisson Blend — `poisson_blend()`

1. Compute mask bounding box → resize replacement to match
2. Place onto full-size canvas at bbox position
3. Gaussian-blur mask edges → threshold → `cv2.seamlessClone`

Uses the precise mask contour so only the object region is modified.

---

## Shared Modules

### AI Harmonization — `harmonize_image()`

Both tabs can call Qwen-Image-Edit-Max to fix rough composites:

1. Encode the stitched image as base64 PNG
2. Send to DashScope `MultiModalConversation` API with instructions to fix lighting, shadows, edges, color, and perspective
3. Download the edited result

120-second timeout. Single API call per harmonization.

### Reinhard Color Transfer

Classic algorithm (Reinhard et al., IEEE CG&A 2001), implemented in `libcom_utils.py`:

1. Convert both foreground and background to CIE Lab color space
2. Compute mean and standard deviation for each Lab channel
3. Linearly transform foreground pixels: `(fg − μ_fg) × (σ_bg / σ_fg) + μ_bg`
4. Convert back to RGB

Zero dependencies beyond NumPy + OpenCV. Always available, no API call needed.

### simOPA Composition Scoring — `opa_scorer.py`

Self-contained Object Placement Assessment model from BCMI Lab:

| Component | Detail |
|-----------|--------|
| Architecture | 4-channel ResNet18 → GAP → Linear(512, 2) |
| Input | Composite image + mask, concatenated to 4 channels, resized to 256×256 |
| Output | Softmax probability of class 1 ("reasonable placement") |
| Weight | `checkpoints/simopa.pth` (~45 MB) |
| Device | CPU (no GPU required) |

No dependency on libcom — model code is extracted directly from the OPA project source.

---

## Blending Modes Comparison

| Mode | Speed | Edge Quality | Color Fidelity | API Call |
|------|-------|-------------|----------------|----------|
| Alpha Blend | Instant | Hard edges | Full foreground colors | No |
| Poisson (NORMAL_CLONE) | Fast | Smooth, gradient-matched | Adapted to background | No |
| Poisson + AI Harmonize | ~30s | Seamless | Re-lit, re-colored | Yes |
| + Reinhard Color Transfer | Fast | Same as blend mode | Foreground matched to bg stats | No |

---

## Architecture Decisions

| Choice | Rationale |
|--------|-----------|
| Two tabs | Different use cases: creative placement vs targeted replacement |
| Separate state per tab (`_s1` / `_s2`) | Independent workflows, no cross-contamination |
| SAM3 for foreground auto-segmentation | Center point is heuristic; SAM3 makes it reliable |
| `cv2.warpAffine` for rotation/scale | Fast, CPU-only, handles mask interpolation correctly |
| Self-contained simOPA | Avoids libcom's broken import chain; works on CPU |
| Reinhard color transfer (built-in) | Always available; no API cost; good baseline |
| Gradio `.queue()` | Background thread execution; API calls don't block UI |

---

## Known Limitations

1. **SAM3 CPU latency**: ~45s load + ~5-10s per inference. Mitigation: preload at startup
2. **Single-shot harmonization**: No iterative refinement; re-run manually if needed
3. **crop_to_object**: Fails on off-center or busy-background foregrounds
4. **simOPA score interpretation**: Trained on OPA dataset; scores are relative, not absolute
5. **DashScope content filters**: Sensitive terms blocked; API quota may be limited
6. **No undo**: Each blend overwrites `_s1["blend_result"]`; save intermediate results manually
