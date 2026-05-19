# OpenSplice — Image Stitching Agent

An AI-powered image editing agent that replaces objects in photos using natural language instructions. Describe what you want to change, and the agent handles the rest — locating objects, generating replacements, blending them in, and polishing the result.

```
python -m image_stitch_agent.main \
    -i "test_images/sheyou.jpg" \
    -p "replace the backpack on the floor with a flower pot"
```

Powered by [OpenWorldSAM](https://github.com/NIOResearch/Open-World-SAM2) (segmentation), [Z-Image-Turbo](https://help.aliyun.com/zh/model-studio/) (generation), [Qwen Vision](https://github.com/QwenLM/Qwen) (planning + review), and [Qwen-Image-Edit](https://help.aliyun.com/zh/model-studio/qwen-image-edit-guide) (harmonization). Orchestrated via [LangGraph](https://github.com/langchain-ai/langgraph).

For the full technical breakdown, see [Method.md](Method.md).

---

## Table of Contents

- [Installation](#installation)
- [Setup](#setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)

---

## Installation

### 1. Prerequisites

- **Python 3.10+**
- **PyTorch 2.x** (CPU or CUDA)

### 2. Clone and create environment

```bash
git clone <repo-url> OpenSplice
cd OpenSplice

# Create and activate a virtual environment
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

### 4. Install detectron2

detectron2 must be built from source to match your PyTorch version.

```bash
# CPU-only (Windows/Linux)
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Or with CUDA (replace cu121 with your CUDA version)
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2'
```

If the git clone is slow, alternative:
```bash
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2 && pip install -e .
```

### 5. Verify installation

```bash
python -c "import torch; import detectron2; import cv2; from transformers import AutoTokenizer; print('All OK')"
```

---

## Setup

### 1. Get a DashScope API Key

1. Go to [bailian.console.alibabacloud.com](https://bailian.console.alibabacloud.com/)
2. Register / log in with your Alibaba Cloud account
3. Navigate to **API Key Management** → create a new API key
4. Copy the key (starts with `sk-`)

### 2. Download OpenWorldSAM model weights

Download the following files and place them in `OpenWorldSAM-main/checkpoints/`:

| File | Size | Download |
|------|------|----------|
| `model_final.pth` | ~1.0 GB | [Google Drive](https://drive.google.com/drive/folders/1bBPR2FzNkCU0rn3noZDCJjqF0mrhgvFu) or [HuggingFace](https://huggingface.co/YxZhang/evf-sam2-multitask) |
| `sam2_hiera_large.pt` | ~856 MB | `python download_sam2_backbone.py` or from [SAM2 repo](https://github.com/facebookresearch/sam2) |

The directory should look like:
```
OpenWorldSAM-main/checkpoints/
├── model_final.pth
└── sam2_hiera_large.pt
```

### 3. Configure `.env`

Copy the template below to `OpenSplice/.env` and fill in your values:

```bash
# ============================================================
# Image Stitch Agent - Configuration
# ============================================================

# --- Qwen Vision API (task decomposition + visual review) ---
QWEN_API_KEY=sk-your-api-key-here
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_VISION_MODEL=qwen3.6-flash

# --- Image Generation API (Z-Image-Turbo) ---
QWEN_IMAGE_API_KEY=sk-your-api-key-here
QWEN_IMAGE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_IMAGE_MODEL=z-image-turbo

# --- OpenWorldSAM paths ---
# Absolute path to the OpenWorldSAM-main directory
OWSAM_REPO_ROOT=D:/path/to/OpenSplice/OpenWorldSAM-main
# Relative paths within OWSAM_REPO_ROOT
OWSAM_CONFIG=configs/refcoco/Open-World-SAM2-CrossAttention.yaml
OWSAM_CHECKPOINT=checkpoints/model_final.pth
OWSAM_SAM2_BACKBONE=checkpoints/sam2_hiera_large.pt
OWSAM_DEVICE=cpu

# --- Detectron2 datasets root ---
# Points to OWSAM's datasets directory (even if empty, needed for metadata registration)
DETECTRON2_DATASETS=D:/path/to/OpenSplice/OpenWorldSAM-main/datasets

# --- Output directory (relative to OpenSplice/) ---
OUTPUT_DIR=./outputs
```

**IMPORTANT**: Use **absolute paths** with forward slashes for `OWSAM_REPO_ROOT` and `DETECTRON2_DATASETS`.

---

## Usage

### Basic

```bash
cd OpenSplice

python -m image_stitch_agent.main \
    -i "test_images/sheyou.jpg" \
    -p "replace the backpack on the floor with a flower pot"
```

### Arguments

| Flag | Required | Description |
|------|----------|-------------|
| `-i`, `--image` | Yes | Path to the input image (absolute or relative) |
| `-p`, `--instruction` | Yes | Natural language editing instruction |
| `-v`, `--verbose` | No | Show debug-level logs |

### Examples

```bash
# Replace an object
python -m image_stitch_agent.main \
    -i "OpenWorldSAM-main/demo/images/dog.jpg" \
    -p "replace the dog with a plush toy of Peppa Pig"

# Change clothing color (via generation)
python -m image_stitch_agent.main \
    -i "test_images/portrait.jpg" \
    -p "change the person's red jacket to a blue denim jacket"

# Add an object to a scene
python -m image_stitch_agent.main \
    -i "test_images/room.jpg" \
    -p "put a large green potted plant next to the desk"
```

### What to expect

The pipeline outputs progress to the terminal:

```
============================================================
  STEP 1: Task Decomposition (Vision LLM)
============================================================
  Plan: 2 steps
    [1] SEGMENT: a black backpack sitting on the wooden floor...
    [2] GENERATE: A realistic flower pot resting on wooden floor...
  Blend: step 2 -> step 1 (poisson)

============================================================
  STEP 2: Segmentation (OpenWorldSAM)
============================================================
  Running 1 segmentation(s)...
    [1] mask area=29656px, score=0.889

============================================================
  STEP 3: Image Generation (Z-Image-Turbo)
============================================================
  [2] Generating image...
       result: 1024x1024

============================================================
  STEP 4: Stitching (Poisson Blend)
============================================================
  Mask bbox: (422, 1046, 189, 229)
  Stitching complete.

============================================================
  STEP 5: Visual Review (Vision LLM)
============================================================
  Approved: True, Score: 8/10

============================================================
  FINAL: Result accepted
============================================================
```

If the review finds visual issues (score < 5), an additional **harmonization** step runs:

```
  Approved: False, Score: 3/10
    - The object looks like a flat sticker pasted on
    - Lighting does not match the scene

  -> Will harmonize (in-place image edit)

============================================================
  STEP 6: Harmonization (Qwen-Image-Edit)
============================================================
  Harmonization complete. Result: 1184x896

============================================================
  STEP 5: Visual Review (Vision LLM)
============================================================
  Approved: True, Score: 7/10

============================================================
  FINAL: Result accepted
============================================================
```

### Total API calls per run

| Call | Model | Purpose |
|------|-------|---------|
| 1 | Qwen Vision | Task decomposition |
| 2 | Z-Image-Turbo | Generate replacement image |
| 3 | Qwen Vision | Visual quality review |
| 4 (optional) | Qwen-Image-Edit | Harmonize if review fails |

Fixed budget of 3–4 calls. No retry loops.

---

## Project Structure

```
OpenSplice/
├── image_stitch_agent/          # Core agent package
│   ├── __init__.py
│   ├── config.py                # Loads .env, exports all settings
│   ├── llm_client.py            # Qwen Vision client (decompose + review)
│   ├── owsam_wrapper.py         # OpenWorldSAM inference wrapper (lazy init)
│   ├── image_gen_client.py      # Image generation + harmonization clients
│   ├── stitcher.py              # Poisson blending (cv2.seamlessClone)
│   ├── workflow.py              # LangGraph state machine (6 nodes + routing)
│   └── main.py                  # CLI entry point (argparse)
├── OpenWorldSAM-main/           # OpenWorldSAM source code
│   ├── checkpoints/             # *** NOT in git — model weights go here ***
│   ├── configs/                 # Model configs (refcoco, coco, ade20k, ...)
│   ├── demo/                    # Inference utilities + demo images
│   ├── model/                   # Model architecture (SAM2 backbone + EVF head)
│   └── utils/                   # Visualizer, constants
├── test_images/                 # Sample input images
├── outputs/                     # *** NOT in git — generated results ***
├── .env                         # *** NOT in git — API keys and paths ***
├── .gitignore
├── download_sam2_backbone.py    # Helper to download SAM2 weights
├── requirements_agent.txt       # Python dependencies
├── test_owsam.py                # OpenWorldSAM smoke test
├── Method.md                    # Full methodology documentation
└── README.md                    # This file
```

---

## How It Works

```
Input Image + Instruction
         │
         ▼
┌─────────────────────┐
│ 1. Decompose         │  Qwen Vision sees the image, plans subtasks
│    Vision LLM        │  → JSON: [{segment: "..."}, {generate: "..."}]
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 2. Segment            │  OpenWorldSAM referring expression segmentation
│    OpenWorldSAM       │  → Binary mask + confidence score + bbox
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 3. Generate           │  Z-Image-Turbo text-to-image
│    Z-Image-Turbo      │  → 1024×1024 replacement image
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 4. Stitch             │  Resize → place at mask bbox → Poisson blend
│    cv2.seamlessClone  │  → Seamless composite image
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 5. Review            │  Qwen Vision compares before/after
│    Vision LLM        │  → Score (1–10) + issues list
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    │           │
  Approved   Rejected
    │           │
    │           ▼
    │   ┌─────────────────────┐
    │   │ 6. Harmonize         │  Qwen-Image-Edit fixes lighting/shadows
    │   │    Image Edit model  │  → Edited composite in-place
    │   └─────────┬───────────┘
    │             │
    │             ▼
    │         Re-review → Always accepted
    │
    ▼
  Final Output
```

---

## Output Files

All generated files go to `outputs/`:

| File | Description |
|------|-------------|
| `final_<timestamp>.png` | The final result (always saved) |
| `last_generated.png` | The most recent Z-Image-Turbo output (debug, overwritten each run) |
| `last_harmonized.png` | The harmonized image (debug, only if harmonization ran) |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'torchscale'`

```bash
pip install torchscale
```

### `ImportError: Using low_cpu_mem_usage=True requires Accelerate`

```bash
pip install accelerate
```

### `AttributeError: 'EvfSam2Model' object has no attribute 'all_tied_weights_keys'`

Your `transformers` version is too new. Install the specific version:

```bash
pip install transformers==4.36.2
```

### `NotImplementedError: Cannot copy out of meta tensor`

PyTorch 2.9 incompatibility with `low_cpu_mem_usage=True`. This has been patched in the bundled `open_world_sam2.py`. If you update OpenWorldSAM source, re-apply the fix: change `low_cpu_mem_usage=True` to `False` in `model/open_world_sam2.py` line ~114.

### `Connection to huggingface.co timed out`

HuggingFace is slow or unreachable from your network. The tokenizer/config files are cached after first load. Set offline mode:

```bash
set HF_HUB_OFFLINE=1     # Windows
export HF_HUB_OFFLINE=1  # Linux/macOS
```

Or use a VPN.

### `Image generation failed: code=InvalidParameter, message=url error`

Your prompt likely triggered DashScope's content safety filter. Rephrase to avoid sensitive terms. For example, "military mortar" → "metal cylinder with a bipod stand", "atomic bomb" → "large grey metal canister".

### `FileNotFoundError: Cannot read image from: ...`

Image paths in the instruction must exist. Use absolute paths or paths relative to `OpenSplice/` (the working directory).

### detectron2 import errors

detectron2 must be installed from source to match your PyTorch version. Reinstall:

```bash
pip uninstall detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### Empty or nonsensical results

The Z-Image-Turbo model sometimes misinterprets ambiguous words (e.g., "mortar" as kitchen tool instead of weapon). Try being more specific in your instruction — the decomposition LLM can generate a better prompt if your instruction is more detailed.

---

## Notes

- **First run is slow**: OpenWorldSAM loads the model on first use (~17 seconds CPU, ~5 seconds CUDA). Subsequent runs in the same session use the cached model.
- **CPU inference**: Segmentation takes ~8 seconds per image on CPU. Set `OWSAM_DEVICE=cuda` in `.env` if you have a GPU.
- **Content filtering**: DashScope applies safety filters to both text prompts and generated images. Military, violent, or sensitive content may be blocked.
- **API costs**: Each run costs 3–4 API calls. Rough pricing (DashScope China): Qwen Vision ~¥0.004/1K tokens, Z-Image-Turbo ~¥0.15/image, Qwen-Image-Edit ~¥0.20/image.
