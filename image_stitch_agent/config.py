"""OpenSplice — configuration loaded from .env."""

import os
import warnings
import logging
from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Pass HF_TOKEN from .env to huggingface_hub for SAM 3 checkpoint download
_hf_token = os.getenv("HF_TOKEN", "")
if _hf_token and "your_" not in _hf_token.lower():
    os.environ["HF_TOKEN"] = _hf_token

# Only go offline if the SAM 3 checkpoint already exists locally
_sam3_ckpt = os.getenv("SAM3_CHECKPOINT", "")
if _sam3_ckpt and _sam3_ckpt != "auto" and Path(_sam3_ckpt).exists():
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Suppress noisy third-party warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def _require(key: str) -> str:
    val = os.getenv(key, "")
    if not val or "your_" in val.lower():
        raise ValueError(f"Missing or placeholder value for {key} in .env")
    return val


# --- DashScope API (Qwen Image Edit for pose adaptation) ---
QWEN_IMAGE_API_KEY = _require("QWEN_IMAGE_API_KEY")
QWEN_IMAGE_MODEL = os.getenv("QWEN_IMAGE_MODEL", "z-image-turbo")
DASHSCOPE_IMAGE_EDIT_MODEL = os.getenv("QWEN_IMAGE_EDIT_MODEL", "qwen-image-edit-max")

# --- SAM 3 ---
_sam3_raw = os.getenv("SAM3_CHECKPOINT", str(_PROJECT_ROOT / "checkpoints" / "sam3.pt"))
if _sam3_raw == "auto":
    SAM3_CHECKPOINT = None  # let SAM 3 auto-download from HuggingFace
else:
    p = Path(_sam3_raw)
    SAM3_CHECKPOINT = str(p.resolve() if not p.is_absolute() else p)
SAM3_DEVICE = os.getenv("SAM3_DEVICE", "cpu")

# --- Output ---
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(_PROJECT_ROOT / "outputs")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
