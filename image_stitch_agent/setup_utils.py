"""
Setup utilities: download SAM2 backbone and verify environment.
"""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def download_sam2_backbone(save_dir: Path) -> Path:
    """
    Download SAM2 hiera_large backbone (~2.4GB).
    Returns path to the downloaded checkpoint.
    """
    url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    save_path = save_dir / "sam2_hiera_large.pt"

    if save_path.exists():
        logger.info("SAM2 backbone already exists at %s", save_path)
        return save_path

    logger.info("Downloading SAM2 backbone from %s ...", url)
    logger.info("File size: ~2.4GB, this may take a while...")

    import requests
    from tqdm import tqdm

    response = requests.get(url, stream=True, timeout=600)
    total = int(response.headers.get("content-length", 0))

    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="sam2_hiera_large.pt"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    logger.info("SAM2 backbone downloaded to %s", save_path)
    return save_path


def check_environment() -> dict:
    """Check if the required environment is set up correctly."""
    issues = []

    # Check PyTorch
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if not cuda_ok:
            issues.append("CUDA not available — OpenWorldSAM will run on CPU (slow)")
    except ImportError:
        issues.append("PyTorch not installed")

    # Check detectron2
    try:
        import detectron2
    except ImportError:
        issues.append(
            "detectron2 not installed. Install with: "
            "pip install --extra-index-url https://miropsota.github.io/torch_packages_builder "
            "detectron2==0.6+2a420edpt2.5.0cu121"
        )

    # Check SAM2 backbone
    from image_stitch_agent.config import OWSAM_REPO_ROOT, OWSAM_SAM2_BACKBONE
    backbone_path = OWSAM_REPO_ROOT / OWSAM_SAM2_BACKBONE
    if not backbone_path.exists():
        issues.append(f"SAM2 backbone not found at {backbone_path}. Run download_sam2_backbone() to get it.")

    # Check OpenWorldSAM checkpoint
    from image_stitch_agent.config import OWSAM_CHECKPOINT
    ckpt_path = OWSAM_REPO_ROOT / OWSAM_CHECKPOINT
    if not ckpt_path.exists():
        issues.append(f"OpenWorldSAM checkpoint not found at {ckpt_path}")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
    }
