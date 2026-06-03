# OpenSplice — AI Image Stitching

Interactive web UI for object replacement in images. Two workflows: **free-form interactive placement** (drag, rotate, scale, blend) and **SAM 3 mask-based stitching** (segment → replace → stitch).

```
python -m image_stitch_agent.app
# → http://127.0.0.1:7860
```

Also includes a standalone SAM 3 segmentation demo:

```
python sam3_demo.py
```

Powered by [SAM 3](https://github.com/facebookresearch/sam3) (segmentation), [Qwen-Image-Plus](https://help.aliyun.com/zh/model-studio/) (generation), [Qwen-Image-Edit](https://help.aliyun.com/zh/model-studio/qwen-image-edit-guide) (harmonization), [simOPA](https://github.com/bcmi/Object-Placement-Assessment) (scoring), and [Reinhard Color Transfer](https://doi.org/10.1109/38.946629) (color matching).

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

The SAM 3 model checkpoint (`sam3.pt`) is **required** (~3.4 GB).

**Option A — ModelScope (recommended for users in China):**

```bash
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('facebook/sam3', cache_dir='checkpoints')
"
# Copy the .pt file to checkpoints/sam3.pt
```

**Option B — HuggingFace:**

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('facebook/sam3', 'sam3.pt', local_dir='checkpoints')
"
```

Expected layout:

```
checkpoints/
├── sam3.pt         (~3.4 GB, SAM 3 model)
└── simopa.pth      (~45 MB, simOPA scorer — auto-copied from project/)
```

---

## Setup

### 1. DashScope API Key

1. Go to [bailian.console.alibabacloud.com](https://bailian.console.alibabacloud.com/)
2. Register / log in with your Alibaba Cloud account
3. Navigate to **API Key Management** → create a new API key
4. Copy the key (starts with `sk-`)

### 2. Configure `.env`

```bash
QWEN_IMAGE_API_KEY=sk-your-api-key-here
QWEN_IMAGE_MODEL=qwen-image-plus          # or z-image-turbo
QWEN_IMAGE_EDIT_MODEL=qwen-image-edit-max # or qwen-image-edit-plus

SAM3_CHECKPOINT=checkpoints/sam3.pt
SAM3_DEVICE=cpu

OUTPUT_DIR=./outputs
```

---

## Usage

### OpenSplice Web UI

```bash
python -m image_stitch_agent.app
# Open http://127.0.0.1:7860
```

Two tabs:

**Tab 1 — Interactive Placement:** free-form drag, rotate, scale

1. Upload background image
2. Upload or AI-generate a foreground image (auto-segmented via SAM 3 center point)
3. Click on the preview canvas to position the foreground
4. Adjust rotation (-180° to 180°) and scale (0.1× to 3.0×) with sliders
5. Blend:
   - **Alpha Blend** — direct pixel copy (fastest, hard edges)
   - **Fast Blend (Poisson)** — gradient-domain seamless cloning
   - **AI Harmonize** — Poisson blend + DashScope AI in-place fix
6. Optional: **Color Transfer (Reinhard)** to match foreground colors to background
7. Optional: **Score Naturalness (simOPA)** to evaluate compositing quality

**Tab 2 — SAM3 Stitch:** segment → replace → stitch

1. Upload image (file or webcam)
2. Segment an object using text prompt, box drag, or point clicks
3. Select mask if multiple results
4. Provide replacement — upload an image or generate via AI
5. Stitch:
   - **Fast Stitch** — Poisson blend with auto object extraction
   - **Pose-Adapt & Stitch** — Fast stitch + AI harmonization

### SAM 3 Demo (segmentation only)

```bash
python sam3_demo.py
```

Supports text, box, and point prompts. No stitching — just segmentation visualization.

---

## Project Structure

```
OpenSplice/
├── image_stitch_agent/
│   ├── __init__.py
│   ├── config.py                # .env loader
│   ├── segmenter.py             # SAM 3 wrapper (text/box/point)
│   ├── stitcher.py              # Poisson blend, crop_to_object
│   ├── transforms.py            # Rotation, scale, overlay, place_and_blend, alpha_place
│   ├── image_gen_client.py      # DashScope APIs (generate, harmonize)
│   ├── libcom_utils.py          # Reinhard color transfer + simOPA bridge
│   ├── opa_scorer.py            # Self-contained simOPA scorer (CPU, no libcom dep)
│   └── app.py                   # Gradio Web UI (two tabs)
├── sam3_demo.py                 # Standalone SAM 3 demo
├── test_segmentation.py         # CLI SAM 3 test
├── checkpoints/
│   ├── sam3.pt                  # SAM 3 weights (~3.4 GB)
│   └── simopa.pth               # simOPA weights (~45 MB)
├── outputs/                     # Result images
├── .env                         # API keys and config
├── requirements_agent.txt
├── README.md
└── Method.md
```

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/last_placement_alpha.png` | Tab 1 Alpha Blend result |
| `outputs/last_placement_fast.png` | Tab 1 Poisson blend result |
| `outputs/last_placement_harmonized.png` | Tab 1 AI harmonized result |
| `outputs/last_placement_colortransfer.png` | Tab 1 Reinhard color transfer result |
| `outputs/last_fast.png` | Tab 2 Fast Stitch result |
| `outputs/last_result.png` | Tab 2 AI-harmonized result |
| `outputs/last_generated.png` | Most recent AI-generated image |

---

## Models

| Model | Where | Purpose |
|-------|-------|---------|
| SAM 3 (848M) | `checkpoints/sam3.pt` | Segmentation (text/box/point) |
| simOPA (~11M) | `checkpoints/simopa.pth` | Composition naturalness scoring |
| Qwen-Image-Plus | DashScope API | AI image generation |
| Qwen-Image-Edit-Max | DashScope API | AI image harmonization |

---

## Distribution

### Option A: PyInstaller Standalone .exe

Build a single `.exe` file (~300 MB). Double-click to run — pretrained weights auto-download on first launch.

```bash
pip install pyinstaller
python build_exe.py
# Output: dist/OpenSplice.exe
```

Share `OpenSplice.exe` + `.env`（with API key）. The recipient needs:
- Windows 10/11
- No Python required
- ~15 GB free disk space (for SAM3 checkpoint + PyTorch extraction)
- Internet connection (first run downloads SAM3 from HuggingFace/ModelScope)

### Option B: HuggingFace Spaces

Free hosting, zero download for users. Deploy to [huggingface.co/spaces](https://huggingface.co/spaces):

1. Create a new Space with Gradio SDK
2. Push the `image_stitch_agent/` folder + `checkpoints/` + `requirements_agent.txt`
3. Set `QWEN_IMAGE_API_KEY` as a Space secret

### Option C: Gradio Share Link

Temporary public link from your machine:

```python
# In app.py, change launch() to:
ui.launch(share=True)
```

Anyone with the link can use your running instance.

---

## Troubleshooting

### SAM 3 checkpoint not found

Make sure `sam3.pt` exists at `checkpoints/sam3.pt`.

### DashScope API errors

- Check `QWEN_IMAGE_API_KEY` in `.env`
- If quota exhausted, switch models in `.env` (e.g., `qwen-image-plus` ↔ `z-image-turbo`)

### Scale/Rotation shows no effect

Fixed — SAM masks are binary (0/1), thresholding now correctly handles this.

### simOPA scoring unavailable

Check `checkpoints/simopa.pth` exists. If not, copy from `project/OPA/eval_opascore/checkpoints/simopa.pth`.

### Slow first run

SAM 3 loads on first use (~45s CPU). Subsequent calls use the cached model. Set `SAM3_DEVICE=cuda` if you have a GPU.
