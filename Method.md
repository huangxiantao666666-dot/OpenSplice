# Method: Interactive Segmentation + Poisson Blending + AI Harmonization

## Overview

OpenSplice provides an **interactive web UI** for object replacement in images. The user segments a target object with SAM 3 (using text, box, or point prompts), provides a replacement image (uploaded or AI-generated), and the system blends it in — with an optional AI harmonization step that fixes lighting, seams, and scale mismatches.

**Core insight**: SAM 3's open-vocabulary segmentation handles arbitrary objects without training; Poisson blending provides gradient-domain seamless insertion; and Qwen-Image-Edit acts as a "fix-it" pass on the rough composite.

---

## Pipeline

```
1. Segmentation       SAM 3 (text / box / point)
        │
2. Replacement        Upload or Qwen-Image-Plus generation
        │
3. Object Extraction  Canny edge detection → crop to subject bbox
        │
4. Poisson Blend      Resize → place at mask bbox → cv2.seamlessClone
        │
5. Harmonization      Qwen-Image-Edit in-place fix (optional)
```

---

## 1. Segmentation — SAM 3

**Model**: SAM 3 (`facebook/sam3`, 848M params) with interactive predictor enabled (`enable_inst_interactivity=True`).

Three prompt modes:

| Mode | Method | Use case |
|------|--------|----------|
| **Text** | `Sam3Processor.set_text_prompt()` → grounding boxes → `model.predict_inst()` per box | "a person", "穿红色衣服的人" |
| **Box** | `model.predict_inst(box=[x1,y1,x2,y2])` | Drag two corners to enclose an object |
| **Point** | `model.predict_inst(point_coords=..., point_labels=...)` | Click foreground points, multi-mask for single click |

Text grounding: SAM 3's `Sam3Processor` detects all instances matching a text concept, returning coarse boxes and masks. Each box is then refined through the interactive predictor (`predict_inst` with box input) to produce high-quality instance masks.

Box and point: go directly through the interactive predictor, which is SAM 1-style point/box → mask inference.

All masks are binary (0/1 uint8), thresholded at 0.5.

**Workaround**: `Sam3Processor.set_image` has a numpy shape bug — for HWC arrays it reads `shape[-2:]` as `(W, C)`. Fixed by converting to PIL Image before passing to the processor.

---

## 2. Replacement Image

Two sources:

- **Upload**: User-provided image file, loaded via `cv2.imread` → converted to RGB
- **Generate**: Qwen-Image-Plus text-to-image via DashScope SDK (`dashscope.aigc.image_generation.ImageGeneration.call()`)

Both are normalized to RGB internally. Generated images are 1024×1024.

---

## 3. Object Extraction — `crop_to_object()`

AI-generated images include background around the subject. Before blending, the subject must be isolated.

**Algorithm** (pure OpenCV, no extra AI calls):

1. Convert to grayscale
2. Canny edge detection (thresholds 30/100)
3. Morphological close (elliptical 7×7 kernel, 2 iterations) to connect edge fragments
4. Find external contours
5. Filter contours near the image center (within 40% of max dimension) — assumes subject is centered
6. Compute combined bounding box of all valid contours + 8px padding
7. Crop

This removes the background ring around AI-generated subjects in milliseconds.

---

## 4. Poisson Blending — `poisson_blend()`

**Method**: `cv2.seamlessClone` with `NORMAL_CLONE` mode.

**Steps**:

1. Compute mask bounding box `(x, y, w, h)` from the SAM 3 mask
2. Resize the replacement to `(w, h)`
3. Place onto a full-size canvas at the bbox position
4. Apply Gaussian blur to the mask (kernel size 5) for edge softening
5. Threshold blurred mask at 0.5 → multiply by 255 to get uint8 mask in [0, 255]
6. Call `cv2.seamlessClone(fg_full, background, mask_uint8, center, NORMAL_CLONE)`

**`NORMAL_CLONE` vs `MIXED_CLONE`**: `NORMAL_CLONE` preserves the source (foreground) texture while smoothly adapting to the destination (background) gradient at the mask boundary. `MIXED_CLONE` blends gradients from both, which creates an "averaged" look — not what we want for object insertion.

**Mask format note**: SAM 3 masks are binary (0/1). The threshold is `> 0.5` (not `> 128`) because the blur operates on float32 values in [0, 1].

**Fallback**: If `seamlessClone` raises an error (e.g., mask region too small), falls back to alpha blending with Gaussian feathering (`_alpha_blend`).

---

## 5. AI Harmonization — `harmonize_image()`

**Model**: Qwen-Image-Edit-Max (DashScope `MultiModalConversation` API)

The Fast Stitch (Poisson blend) result is a rough composite — edges may be visible, lighting may mismatch, scale may look wrong. Harmonization sends this composite to the image editing model with instructions to:

1. Fix lighting and shadows to match the scene
2. Blend edges seamlessly (no visible seams or halos)
3. Match color tone and white balance to the background
4. Adjust scale and perspective if the object looks disproportionate
5. **Keep the background and non-pasted regions identical**

The stitched image is sent as base64 PNG alongside the text prompt. The model returns an edited image where only the pasted region and its immediate surroundings are adjusted.

**API call budget**: 1 call per harmonization. 120-second timeout. No retry loops.

---

## Architecture Decisions

| Choice | Rationale |
|--------|-----------|
| SAM 3 | State-of-the-art open-vocabulary segmentation; supports text, box, and point prompts; Chinese + English |
| Poisson blending | Gradient-domain blending is smoother than alpha blending, faster than deep inpainting, offline-capable |
| `NORMAL_CLONE` | Preserves replacement texture while adapting to background lighting |
| Canny edge detection for object extraction | Fast (milliseconds), no extra AI call, works for centered subjects |
| Qwen-Image-Edit for harmonization | In-place editing preserves object position/shape; 1 call vs unpredictable retry loops |
| Gradio Web UI | Interactive prompt adjustments; immediate visual feedback; webcam support |
| Single DashScope API key | One provider for generation + editing; no VPN needed for Chinese users |

---

## Known Limitations

1. **Generation quality**: AI-generated replacements may differ in style, lighting, or proportion from the original scene
2. **Content safety filters**: Sensitive terms are blocked by DashScope content moderation
3. **CPU inference latency**: SAM 3 loads in ~45s and infers in ~5-10s on CPU
4. **Single-shot harmonization**: No iterative refinement; if the first harmonization doesn't look right, re-run manually
5. **crop_to_object edge cases**: If the replacement subject is off-center or has a complex background, edge detection may include background in the crop
6. **Mask-dependent quality**: The final blend is only as good as the SAM 3 mask; inaccurate masks produce visible artifacts at the boundary
