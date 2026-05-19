# OpenSplice — Image Stitching Agent

An AI-powered image editing agent that replaces objects in photos using natural language instructions. Powered by [OpenWorldSAM](https://github.com/NIOResearch/Open-World-SAM2), [Qwen Vision](https://github.com/QwenLM/Qwen), and [Z-Image-Turbo](https://help.aliyun.com/zh/model-studio/), orchestrated via [LangGraph](https://github.com/langchain-ai/langgraph).

## Quick Start

```bash
cd OpenSplice

# Install dependencies
pip install -r requirements_agent.txt

# Configure API keys and model paths
# Edit .env with your settings

# Run the agent
python -m image_stitch_agent.main \
    -i "test_images/sheyou.jpg" \
    -p "replace the backpack on the floor with a flower pot"
```

## Requirements

- Python 3.10+
- PyTorch 2.x (CPU or CUDA)
- [detectron2](https://github.com/facebookresearch/detectron2) (custom build for your CUDA/PyTorch version)
- OpenWorldSAM weights ([download](https://github.com/NIOResearch/Open-World-SAM2))
- DashScope API key ([get one](https://bailian.console.alibabacloud.com/))

## Project Structure

```
OpenSplice/
├── image_stitch_agent/          # Core package
│   ├── config.py                # Configuration from .env
│   ├── llm_client.py            # Qwen Vision client (decompose + review)
│   ├── owsam_wrapper.py         # OpenWorldSAM inference wrapper
│   ├── image_gen_client.py      # Image generation + harmonization
│   ├── stitcher.py              # Poisson blending utilities
│   ├── workflow.py              # LangGraph state machine
│   └── main.py                  # CLI entry point
├── OpenWorldSAM-main/           # OpenWorldSAM model code + weights
├── test_images/                 # Input images
├── outputs/                     # Generated results
├── .env                         # API keys and paths
├── test_owsam.py                # OpenWorldSAM test script
├── Method.md                    # Detailed methodology
└── README.md                    # This file
```

## How It Works

1. **Decompose** — Vision LLM looks at the image and plans: what to segment, what to generate
2. **Segment** — OpenWorldSAM locates the target object via referring expression
3. **Generate** — Z-Image-Turbo creates the replacement content
4. **Stitch** — Poisson blending seamlessly pastes the replacement into the original
5. **Review** — Vision LLM compares before/after and scores the result
6. **Harmonize** — If the review fails, Qwen-Image-Edit fixes lighting/shadows in-place

See [Method.md](Method.md) for the full technical breakdown.

## Configuration

Copy `.env` and fill in your settings:

```bash
# Qwen Vision API
QWEN_API_KEY=sk-your-key
QWEN_VISION_MODEL=qwen3.6-flash

# Image Generation API
QWEN_IMAGE_API_KEY=sk-your-key
QWEN_IMAGE_MODEL=z-image-turbo

# OpenWorldSAM paths
OWSAM_REPO_ROOT=D:/path/to/OpenWorldSAM-main
OWSAM_CONFIG=configs/refcoco/Open-World-SAM2-CrossAttention.yaml
OWSAM_CHECKPOINT=checkpoints/model_final.pth
OWSAM_SAM2_BACKBONE=checkpoints/sam2_hiera_large.pt
OWSAM_DEVICE=cpu
```

## CLI Usage

```bash
# Basic usage
python -m image_stitch_agent.main -i <image_path> -p "<instruction>"

# Verbose mode (debug logging)
python -m image_stitch_agent.main -i <image_path> -p "<instruction>" -v

# Examples
python -m image_stitch_agent.main \
    -i "test_images/dog.jpg" \
    -p "replace the dog with a plush toy of Peppa Pig"

python -m image_stitch_agent.main \
    -i "test_images/sheyou.jpg" \
    -p "remove the backpack from the floor"

python -m image_stitch_agent.main \
    -i "test_images/room.jpg" \
    -p "change the red chair into a wooden stool"
```

## Output

Results are saved to `outputs/`:
- `final_<timestamp>.png` — the final result
- `last_generated.png` — the most recent generated image (debug)
- `last_harmonized.png` — the harmonized image (debug, if applicable)

## Test OpenWorldSAM

```bash
python test_owsam.py
```

This verifies that OpenWorldSAM loads correctly and runs inference on a demo image.

## Notes

- OpenWorldSAM runs on CPU by default (set `OWSAM_DEVICE=cuda` if you have a GPU)
- The first run loads the model (~17 seconds CPU, cached thereafter)
- Image generation may be blocked by content safety filters for sensitive terms
- Use `HF_HUB_OFFLINE=1` if HuggingFace is slow or unreachable
