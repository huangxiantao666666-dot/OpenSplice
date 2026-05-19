import os
import warnings
import logging
from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Suppress noisy third-party warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*non-writable.*", module="torchvision")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("fvcore").setLevel(logging.ERROR)
logging.getLogger("detectron2").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def _require(key: str) -> str:
    val = os.getenv(key, "")
    if not val or "your_" in val.lower():
        raise ValueError(f"Missing or placeholder value for {key} in .env")
    return val


# --- Qwen Vision (used for both task decomposition & visual review) ---
QWEN_API_KEY = _require("QWEN_API_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_VISION_MODEL = os.getenv("QWEN_VISION_MODEL", "qwen3.6-flash")

# --- Qwen Image Gen ---
QWEN_IMAGE_API_KEY = os.getenv("QWEN_IMAGE_API_KEY", QWEN_API_KEY)
QWEN_IMAGE_BASE_URL = os.getenv(
    "QWEN_IMAGE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
QWEN_IMAGE_MODEL = os.getenv("QWEN_IMAGE_MODEL", "z-image-turbo")

# --- OpenWorldSAM ---
OWSAM_REPO_ROOT = Path(os.getenv("OWSAM_REPO_ROOT", ""))
OWSAM_CONFIG = os.getenv("OWSAM_CONFIG", "configs/refcoco/Open-World-SAM2-CrossAttention.yaml")
OWSAM_CHECKPOINT = os.getenv("OWSAM_CHECKPOINT", "checkpoints/model_final.pth")
OWSAM_SAM2_BACKBONE = os.getenv("OWSAM_SAM2_BACKBONE", "checkpoints/sam2_hiera_large.pt")
OWSAM_DEVICE = os.getenv("OWSAM_DEVICE", "cuda")

# Set DETECTRON2_DATASETS env var (needed before importing OpenWorldSAM datasets)
_d2_datasets = os.getenv("DETECTRON2_DATASETS", "")
DETECTRON2_DATASETS = _d2_datasets
if _d2_datasets:
    os.environ["DETECTRON2_DATASETS"] = _d2_datasets

# --- Output ---
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(_PROJECT_ROOT / "outputs")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- System prompt for task decomposition (vision model sees the image) ---
SYSTEM_PROMPT = """You are an image editing task planner. You will receive:
1. An **image** — look at it carefully to understand the scene: what objects are present, their positions, colors, sizes, lighting conditions, and camera angle.
2. A **user instruction** — what they want to change about the image.

Your job is to decompose the user's request into precise subtasks.

You have access to these tools:
1. **OpenWorldSAM**: An open-vocabulary segmentation model. Give it a natural language referring expression describing an object/region in the image, and it returns a precise binary mask. Use concrete, visually grounded descriptions based on what you actually SEE in the image.
2. **Image Generator (Z-Image-Turbo)**: A text-to-image model. Give it a detailed visual description of what to generate, and it produces an image. Describe the desired output with attention to lighting, angle, and style so it matches the original scene.
3. **User-provided images**: If the user specifies their own image file path for replacement, use that instead of generating.

For each user request, output a JSON plan with this structure:
{
  "steps": [
    {
      "step_id": 1,
      "action": "segment",
      "target_description": "concrete visual description of the region to locate, based on what you see in the image",
      "purpose": "what this mask will be used for"
    },
    {
      "step_id": 2,
      "action": "generate",
      "generation_prompt": "detailed visual prompt for the text-to-image model, with lighting/angle/style notes matching the original scene",
      "source_image_path": null,
      "reference_mask_step": 1,
      "purpose": "what to generate and why"
    }
  ],
  "final_placement": {
    "paste_region_step": 1,
    "source_step": 2,
    "blend_mode": "poisson"
  }
}

CRITICAL RULES:
- **LOOK at the image first.** Your target_description must reference what you actually see — not generic descriptions. For example, don't say "the person wearing red clothes" if you can see it's actually "the person in a bright crimson hoodie standing on the left side of the frame."
- "action" must be one of: "segment", "generate"
- "segment" steps produce masks
- "generate" steps: set "source_image_path" to the file path (and leave "generation_prompt" empty) if using a user-provided image
- Generation prompts MUST describe the desired output in context of the original scene: match lighting direction, color temperature, camera angle, and image style (photorealistic/illustration/etc.)
- Mask descriptions should be precise enough for a segmentation model to uniquely identify the target
- Output ONLY valid JSON, no markdown fences or extra text.
"""
