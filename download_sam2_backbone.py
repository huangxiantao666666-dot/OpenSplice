#!/usr/bin/env python
"""
Download SAM2 backbone for OpenWorldSAM.
Run this once before using the image stitch agent.

Usage:
    python download_sam2_backbone.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from image_stitch_agent.setup_utils import download_sam2_backbone
from image_stitch_agent.config import OWSAM_REPO_ROOT

if __name__ == "__main__":
    save_dir = OWSAM_REPO_ROOT / "checkpoints"
    path = download_sam2_backbone(save_dir)
    print(f"SAM2 backbone saved to: {path}")
