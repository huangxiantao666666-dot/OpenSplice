# Method: Vision-Language Driven Image Stitching Agent

## Overview

An **image editing agent powered by vision-language models** that accepts natural language instructions. The agent understands the user's intent, locates target regions via open-vocabulary segmentation, generates replacement content, seamlessly blends it into the original image, and optionally harmonizes in-place — all without manual mask drawing or parameter tuning.

**Core insight**: Chaining open-vocabulary segmentation, text-to-image generation, and instruction-based image editing into a closed-loop feedback system, with a vision model acting as both planner and quality inspector.

---

## Workflow

```
User Instruction + Original Image
      │
      ▼
┌──────────────────────────┐
│ 1. Task Decomposition    │  Qwen Vision (qwen3.6-flash)
│    Vision LLM sees image │  → JSON task plan
│    and plans subtasks    │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ 2. Segmentation          │  OpenWorldSAM (NeurIPS 2025)
│    Referring expression  │  → Binary mask + bbox + score
│    locates target object │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ 3. Image Generation      │  Z-Image-Turbo (DashScope)
│    Text-to-image for     │  → 1024×1024 replacement image
│    replacement content   │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ 4. Stitching             │  cv2.seamlessClone (Poisson)
│    Resize → Position     │  → Seamless composite image
│    → Poisson blend       │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ 5. Visual Review         │  Qwen Vision (qwen3.6-flash)
│    Before/after comparison│  → Score (1-10) + issues
│    with vision model     │     + approved / rejected
└──────┬───────────────────┘
       │
       ├── Approved ──→ Output final result
       │
       └── Rejected
              │
              ▼
       ┌──────────────────────────┐
       │ 6. Harmonization         │  Qwen-Image-Edit-Plus
       │    In-place edits on the │  → Harmonized composite
       │    stitched image        │
       └──────────┬───────────────┘
                  │
                  ▼
           Re-review → Output final result
```

**Fixed API call budget**: 3–4 calls per run (decomposition + generation + review + optional harmonization). No retry loops, no regeneration spirals.

---

## Detailed Methodology

### 1. Task Decomposition

**Model**: Qwen Vision (`qwen3.6-flash`)  
**Input**: Original image (base64) + user instruction (natural language)  
**Output**: Structured JSON task plan

```json
{
  "steps": [
    {"step_id": 1, "action": "segment", "target_description": "..."},
    {"step_id": 2, "action": "generate", "generation_prompt": "..."}
  ],
  "final_placement": {
    "paste_region_step": 1,
    "source_step": 2,
    "blend_mode": "poisson"
  }
}
```

**Why a vision model instead of a text-only LLM**: Scene context — object positions, colors, lighting, camera angle — can only be understood by *seeing* the image. A text-only model must guess, leading to inaccurate target descriptions that cause downstream segmentation failures or content mismatches.

The system prompt enforces three rules:
1. **Look first, then plan** — target descriptions must reference what is actually visible in the image
2. **Scene-aware generation** — generation prompts must include lighting direction, color temperature, camera angle, and image style
3. **Valid JSON only** — no markdown fences, no commentary

### 2. Segmentation

**Model**: OpenWorldSAM (`Open-World-SAM2-CrossAttention`)

An extension of SAM2 published at NeurIPS 2025, supporting four modes: instance segmentation, semantic segmentation, panoptic segmentation, and **referring expression segmentation**. We use the referring mode exclusively.

**Input**: Original image + natural language expression (e.g., "a black backpack sitting on the wooden floor near the bottom rungs of the black metal ladder")

**Output**: Binary mask, confidence score, and bounding box:

```
mask area = 29656 px
score      = 0.889
bbox       = (422, 1046, 189, 229)
```

**Advantage over alternatives**: No predefined categories needed. No bounding box or point annotation needed. Any visible object can be located with free-form natural language, including objects not in standard detection datasets.

Environment: PyTorch 2.9 CPU-only. Model loading takes ~17 seconds, single inference takes ~8 seconds.

### 3. Image Generation

**Model**: Z-Image-Turbo (via DashScope Python SDK)

Uses `dashscope.aigc.image_generation.ImageGeneration.call()` in OpenAI-compatible format. Outputs a 1024×1024 pixel image at a single URL.

**Design decisions**:
- Sync call preferred (~4 seconds); falls back to async polling if sync fails
- Images returned as OSS URLs, downloaded and decoded into BGR numpy arrays
- Each generation overwrites `last_generated.png` to avoid file accumulation

### 4. Stitching (Poisson Blending)

**Method**: Poisson image editing (`cv2.seamlessClone`) with adaptive sizing.

**Steps**:
1. Compute bounding box `(x, y, w, h)` from the segmentation mask
2. Center-crop the 1024×1024 generated image to match the bbox aspect ratio
3. Resize the cropped result to exactly `(w, h)`
4. Place the resized foreground onto a full-size canvas at the bbox position
5. Create a rectangular mask matching the bbox area
6. Apply `cv2.seamlessClone` for gradient-domain blending

**Why a rectangular mask instead of the precise segmentation mask**: The generated object (e.g., Peppa Pig) rarely matches the exact silhouette of the original object being replaced (e.g., a dog). Using the precise mask would crop through the generated content in unpredictable ways. A rectangular mask preserves the full generated content and lets Poisson blending naturally smooth the edges within the bbox region.

Three blending modes are supported:
- `NORMAL_CLONE` (default): Preserves foreground texture, transfers background gradient
- `MIXED_CLONE`: Mixes source and destination gradients
- `MONOCHROME_TRANSFER`: Transfers only color and lighting (not texture)

### 5. Visual Review

**Model**: Qwen Vision (`qwen3.6-flash`)

The original image and the stitched composite are sent side-by-side to the vision model. It acts as a quality inspector, evaluating:

- **Lighting consistency** — does the pasted object match the scene's illumination?
- **Shadow plausibility** — are there realistic contact shadows?
- **Edge quality** — are there halos, artifacts, or cutout effects?
- **Perspective matching** — does the object's angle match the camera viewpoint?
- **Object correctness** — is the generated content what the user actually asked for?

Output:
```json
{
  "approved": false,
  "score": 4,
  "issues": [
    "Poor blending: object looks like a flat 2D sticker",
    "Incorrect lighting: character is brightly lit while scene uses flash"
  ],
  "feedback": "The edit correctly identifies the target, but execution is poor...",
  "new_generation_prompt": "..."
}
```

### 6. Harmonization

**Model**: Qwen-Image-Edit-Plus (DashScope `MultiModalConversation` API)

When the visual review rejects the result, instead of regenerating from scratch (which changes object shape and position), we perform **in-place editing** on the stitched composite.

The stitched image (base64) + a repair instruction are sent to the image editing model, which attempts to fix:
- Lighting mismatches
- Missing or incorrect shadows
- Edge halos and artifacts
- Color inconsistencies

**Why this replaces multi-retry loops**:
- Local fixes preserve object position and shape
- The editing model sees the full context and makes context-aware adjustments
- Exactly 1 extra API call, not an unpredictable N-retry loop
- The harmonized result is reviewed once more, then always accepted

---

## Architecture Decisions

| Choice | Rationale |
|--------|-----------|
| LangGraph StateGraph | Multi-node pipeline with conditional routing and feedback loops is cleaner as a graph than nested if-else chains |
| Qwen ecosystem (Vision + Image + Edit) | Single DashScope API key, no VPN needed for Chinese users, consistent SDK |
| OpenWorldSAM | State-of-the-art open-vocabulary segmentation with referring expression support; open-source |
| Poisson blending | Smoother than direct pixel copy, faster than deep inpainting, offline-capable |
| Single-pass harmonization | More cost-effective than multi-round generation retries; more predictable results |
| CPU inference | User's hardware constraint; acceptable for prototype (17s + 8s per image) |

---

## Known Limitations

1. **Generation quality for rare objects**: Z-Image-Turbo struggles with uncommon objects (mortar weapons, atomic bombs), often producing indistinct shapes or lexical ambiguities (mortar → kitchen tool)
2. **Content safety filters**: Military/sensitive terms are blocked by DashScope content moderation
3. **Harmonization may over-correct**: Qwen-Image-Edit can alter the entire image rather than just the pasted region
4. **Fixed output resolution**: Z-Image-Turbo outputs 1024×1024 only; cropping/upscaling needed for high-res originals
5. **CPU inference latency**: OpenWorldSAM loads in ~17s and infers in ~8s on CPU; not suitable for real-time use
6. **No interactive refinement**: Currently a single-shot pipeline; no support for "move it a bit to the left" style iterative feedback
7. **Merge conflicts with original objects**: If the generated object and original background content both appear around the mask boundary, seam artifacts may persist
