#!/usr/bin/env python
"""
Quick test to verify OpenWorldSAM can load and run inference.
Uses referring expression segmentation on demo images.

Usage:
    cd OpenSplice
    python test_owsam.py
"""

import sys
from pathlib import Path

# Ensure OpenSplice/ is on sys.path
_PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT))

# ---------------------------------------------------------------------------
# Step 1: Check environment
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 1: Environment check")
print("=" * 60)

import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

try:
    import detectron2
    print(f"detectron2: OK ({detectron2.__version__ if hasattr(detectron2, '__version__') else 'installed'})")
except ImportError:
    print("detectron2: MISSING — install with:")
    print("  pip install detectron2 (or the appropriate custom build)")
    sys.exit(1)

try:
    import cv2
    print(f"OpenCV: {cv2.__version__}")
except ImportError:
    print("OpenCV: MISSING")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
    print("transformers: OK")
except ImportError:
    print("transformers: MISSING")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 2: Load .env config
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 2: Load configuration")
print("=" * 60)

from image_stitch_agent.config import (
    OWSAM_REPO_ROOT,
    OWSAM_CONFIG,
    OWSAM_CHECKPOINT,
    OWSAM_SAM2_BACKBONE,
    OWSAM_DEVICE,
    DETECTRON2_DATASETS,
)

# Ensure DETECTRON2_DATASETS is set
import os
os.environ["DETECTRON2_DATASETS"] = os.getenv("DETECTRON2_DATASETS", str(OWSAM_REPO_ROOT / "datasets"))

print(f"OWSAM repo: {OWSAM_REPO_ROOT}")
print(f"OWSAM config: {OWSAM_CONFIG}")
print(f"OWSAM checkpoint: {OWSAM_REPO_ROOT / OWSAM_CHECKPOINT}")
print(f"SAM2 backbone: {OWSAM_REPO_ROOT / OWSAM_SAM2_BACKBONE}")
print(f"Device: {OWSAM_DEVICE}")
print(f"DETECTRON2_DATASETS: {os.environ['DETECTRON2_DATASETS']}")

# Check files exist
checkpoint_path = OWSAM_REPO_ROOT / OWSAM_CHECKPOINT
if not checkpoint_path.exists():
    print(f"ERROR: Checkpoint not found at {checkpoint_path}")
    sys.exit(1)
print(f"Checkpoint: OK ({checkpoint_path.stat().st_size / 1e9:.1f} GB)")

backbone_path = OWSAM_REPO_ROOT / OWSAM_SAM2_BACKBONE
if not backbone_path.exists():
    print(f"ERROR: SAM2 backbone not found at {backbone_path}")
    print("Download it with: python download_sam2_backbone.py")
    sys.exit(1)
print(f"SAM2 backbone: OK ({backbone_path.stat().st_size / 1e9:.1f} GB)")

# ---------------------------------------------------------------------------
# Step 3: Add OWSAM repo to path and import model utilities
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 3: Import OpenWorldSAM modules")
print("=" * 60)

owsam_root = str(OWSAM_REPO_ROOT.resolve())
if owsam_root not in sys.path:
    sys.path.insert(0, owsam_root)

# The demo imports
try:
    from demo.inference_utils import (
        setup_cfg,
        load_model,
        prepare_image_inputs,
        build_inference_inputs,
    )
    print("inference_utils: OK")
except Exception as e:
    print(f"inference_utils import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from utils.visualizer import SegmentationResultVisualizer
    print("visualizer: OK")
except Exception as e:
    print(f"visualizer import failed: {e}")
    sys.exit(1)

# Check datasets import
try:
    import datasets  # noqa: F401 — registers dataset metadata
    print("datasets registration: OK")
except Exception as e:
    print(f"datasets import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 4: Load model
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 4: Load OpenWorldSAM model")
print("=" * 60)

# Model code uses relative paths internally — cd to repo root
import os as _os
_orig_cwd = _os.getcwd()
_os.chdir(str(OWSAM_REPO_ROOT))

cfg = setup_cfg(
    str(OWSAM_REPO_ROOT / OWSAM_CONFIG),
    weights=str(checkpoint_path),
    device=OWSAM_DEVICE,
)
cfg.MODEL.OpenWorldSAM2.TEST.INSTANCE_ON = False
cfg.MODEL.OpenWorldSAM2.TEST.SEMANTIC_ON = False
cfg.MODEL.OpenWorldSAM2.TEST.PANOPTIC_ON = False
cfg.MODEL.OpenWorldSAM2.TEST.REFER_ON = True
cfg.MODEL.OpenWorldSAM2.TEST.NMS_ON = False
cfg.MODEL.OpenWorldSAM2.TEST.TOP_K_ON = False

print("Building model...")
model = load_model(cfg)
print("Model loaded successfully!")

# ---------------------------------------------------------------------------
# Step 5: Test inference on a demo image
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 5: Test inference")
print("=" * 60)

# Pick a test image
test_image_path = Path(__file__).parent / "test_images" / "classroom.jpg"
if not test_image_path.exists():
    print(f"Test image not found: {test_image_path}")
    sys.exit(1)

print(f"Test image: {test_image_path}")

# Prepare image
image_bgr, sam_tensor, beit_tensor, height, width = prepare_image_inputs(
    str(test_image_path), cfg.INPUT.FORMAT
)
print(f"Image size: {width}x{height}")

# Run referring expression segmentation
prompts = [
    "person in blue T-shirt",
    "mouse",
    "a bottle of water",
]
category_ids = list(range(len(prompts)))

inputs = build_inference_inputs(
    sam_tensor, beit_tensor, height, width, prompts, category_ids
)

print(f"Running inference with prompts: {prompts}")
import time
t0 = time.time()

with torch.no_grad():
    outputs = model(inputs)[0]

elapsed = time.time() - t0
print(f"Inference completed in {elapsed:.1f}s")

grounding_masks = outputs.get("grounding_mask")
grounding_scores = outputs.get("grounding_scores")

if grounding_masks is None:
    print("ERROR: No masks returned!")
    sys.exit(1)

print(f"Masks shape: {grounding_masks.shape}")
print(f"Scores: {grounding_scores}")

# ---------------------------------------------------------------------------
# Step 6: Save visualization
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 6: Save results")
print("=" * 60)

output_dir = _PROJECT / "outputs"
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "classroom.png"

visualizer = SegmentationResultVisualizer(
    metadata=None, input_format=cfg.INPUT.FORMAT
)
visualizer.save_referring_result(
    image_bgr, grounding_masks, prompts, str(output_path), scores=grounding_scores
)
print(f"Result saved to: {output_path}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
