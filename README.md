# OpenSplice — AI Image Stitching

Interactive web UI for object replacement in images. Segment with SAM 3 (text / box / point), provide a replacement image (upload or AI-generated), and blend it in seamlessly — with optional AI harmonization.

```
python -m image_stitch_agent.app
# → http://127.0.0.1:7860
```

Also includes a standalone SAM 3 segmentation demo:

```
python sam3_demo.py
```

Powered by [SAM 3](https://github.com/facebookresearch/sam3) (segmentation), [Qwen-Image-Plus](https://help.aliyun.com/zh/model-studio/) (generation), and [Qwen-Image-Edit](https://help.aliyun.com/zh/model-studio/qwen-image-edit-guide) (harmonization).

---

## Installation

### 1. Prerequisites

- Python 3.10+
- PyTorch 2.x (CPU or CUDA)

### 2. Create environment

```bash
cd OpenSplice
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements_agent.txt
```

### 4. Download SAM 3 checkpoint

The SAM 3 model checkpoint (`sam3.pt`) is **required** and **not included** in the repo.

**Option A — ModelScope (recommended for users in China):**

```bash
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('facebook/sam3', cache_dir='checkpoints')
"
# Then copy the .pt file to checkpoints/sam3.pt
```

**Option B — HuggingFace:**

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('facebook/sam3', 'sam3.pt', local_dir='checkpoints')
"
```

The directory should look like:

```
checkpoints/
└── sam3.pt    (~3.4 GB)
```

---

## Setup

### 1. DashScope API Key

1. Go to [bailian.console.alibabacloud.com](https://bailian.console.alibabacloud.com/)
2. Register / log in with your Alibaba Cloud account
3. Navigate to **API Key Management** → create a new API key
4. Copy the key (starts with `sk-`)

### 2. Configure `.env`

Edit `OpenSplice/.env`:

```bash
QWEN_IMAGE_API_KEY=sk-your-api-key-here
QWEN_IMAGE_MODEL=qwen-image-plus
QWEN_IMAGE_EDIT_MODEL=qwen-image-edit-max

SAM3_CHECKPOINT=checkpoints/sam3.pt
SAM3_DEVICE=cpu

OUTPUT_DIR=./outputs
```

---

## Usage

### OpenSplice Web UI (full stitching)

```bash
python -m image_stitch_agent.app
# Open http://127.0.0.1:7860
```

**Workflow:**

1. **Upload image** (file or webcam)
2. **Segment** an object using text prompt, box drag, or point clicks
3. **Select mask** if multiple results
4. **Provide replacement** — upload an image or generate one via AI
5. **Stitch:**
   - **Fast Stitch** — Poisson blend only (local, no API)
   - **Pose-Adapt & Stitch** — Poisson blend + AI harmonization (calls API)

### SAM 3 Demo (segmentation only)

```bash
python sam3_demo.py
# Open http://127.0.0.1:7860
```

Supports text, box, and point prompts. No stitching — just segmentation visualization.

---

## Project Structure

```
OpenSplice/
├── image_stitch_agent/          # Main package
│   ├── __init__.py
│   ├── config.py                # Loads .env settings
│   ├── segmenter.py             # SAM 3 wrapper (text/box/point)
│   ├── stitcher.py              # Poisson blending + crop_to_object
│   ├── image_gen_client.py      # DashScope APIs (generate, harmonize)
│   └── app.py                   # Gradio Web UI
├── sam3_demo.py                 # Standalone SAM 3 segmentation demo
├── checkpoints/                 # SAM 3 model weights (not in git)
│   └── sam3.pt
├── outputs/                     # Generated results (not in git)
├── .env                         # API keys and config (not in git)
├── .gitignore
├── requirements_agent.txt       # Python dependencies
└── README.md
```

---

## How It Works

```
Input Image
    │
    ▼
┌─────────────────────┐
│ 1. Segment           │  SAM 3 — text / box / point prompts
│    SAM 3             │  → Binary mask + score + bbox
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 2. Replacement       │  Upload image OR generate via Qwen-Image-Plus
│    Upload / Generate │  → RGB image
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 3. Extract Object    │  Canny edge detection + contours
│    crop_to_object()  │  → Cropped to subject (no background)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 4. Poisson Blend     │  Resize → Place at mask bbox → seamlessClone
│    cv2.seamlessClone │  → Rough composite (Fast Stitch output)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 5. Harmonize (opt.)  │  Qwen-Image-Edit fixes lighting/seams/scale
│    AI Image Edit     │  → Natural-looking final composite
└─────────────────────┘
```

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/last_fast.png` | Fast Stitch result (Poisson blend only) |
| `outputs/last_result.png` | AI-harmonized result (Pose-Adapt & Stitch) |
| `outputs/last_generated.png` | Most recent AI-generated replacement |
| `outputs/last_harmonized.png` | Debug: raw harmonization API output |
| `outputs/last_pose_adapted.png` | Debug: raw pose adaptation API output |

---

## Troubleshooting

### SAM 3 checkpoint not found

Make sure `sam3.pt` exists at `checkpoints/sam3.pt`. See [Download SAM 3 checkpoint](#4-download-sam-3-checkpoint).

### DashScope API errors

- Check `QWEN_IMAGE_API_KEY` in `.env` is correct
- Check model quota — switch models in `.env` if quota exhausted (`qwen-image-plus` ↔ `z-image-turbo`, `qwen-image-edit-max` ↔ `qwen-image-edit-plus`)

### Stitch result looks unchanged

The mask threshold was fixed. Make sure you're running the latest code — older versions had `mask_blurred > 128` instead of `mask_blurred > 0.5`.

### Slow first segmentation

SAM 3 loads on first use (~45s CPU). Subsequent segmentations use the cached model. Set `SAM3_DEVICE=cuda` in `.env` if you have a GPU.

### Content safety filters

DashScope applies safety filters. Rephrase prompts to avoid sensitive terms if generation is blocked.
