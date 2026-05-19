"""
OpenWorldSAM inference wrapper.
Wraps the referring expression segmentation mode for easy use.
"""

import sys
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch

from .config import (
    OWSAM_REPO_ROOT,
    OWSAM_CONFIG,
    OWSAM_CHECKPOINT,
    OWSAM_SAM2_BACKBONE,
    OWSAM_DEVICE,
)

logger = logging.getLogger(__name__)

# Ensure OpenWorldSAM repo is on sys.path
_repo_root = str(OWSAM_REPO_ROOT.resolve())
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Lazy-loaded model handle
_model = None
_cfg = None


def _ensure_model():
    global _model, _cfg

    if _model is not None:
        return _model, _cfg

    import os as _os
    from demo.inference_utils import setup_cfg, load_model

    # Use cached model files (avoids ~7 min HuggingFace online check)
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")

    config_path = str(OWSAM_REPO_ROOT / OWSAM_CONFIG)
    checkpoint_path = str(OWSAM_REPO_ROOT / OWSAM_CHECKPOINT)

    cfg = setup_cfg(config_path, weights=checkpoint_path, device=OWSAM_DEVICE)
    cfg.MODEL.OpenWorldSAM2.TEST.INSTANCE_ON = False
    cfg.MODEL.OpenWorldSAM2.TEST.SEMANTIC_ON = False
    cfg.MODEL.OpenWorldSAM2.TEST.PANOPTIC_ON = False
    cfg.MODEL.OpenWorldSAM2.TEST.REFER_ON = True
    cfg.MODEL.OpenWorldSAM2.TEST.NMS_ON = False
    cfg.MODEL.OpenWorldSAM2.TEST.TOP_K_ON = False

    # Model code uses relative paths internally — must run from repo root
    _orig_cwd = _os.getcwd()
    _os.chdir(str(OWSAM_REPO_ROOT))

    logger.info("Loading OpenWorldSAM model...")
    try:
        model = load_model(cfg)
    finally:
        _os.chdir(_orig_cwd)

    logger.info("OpenWorldSAM model loaded successfully.")

    _model = model
    _cfg = cfg
    return model, cfg


def segment(
    image: np.ndarray,
    prompts: List[str],
    conf_threshold: float = 0.3,
) -> List[dict]:
    """
    Run referring expression segmentation on an image.

    Args:
        image: BGR image as numpy array (H, W, 3), uint8.
        prompts: List of natural language referring expressions.
        conf_threshold: Minimum confidence score to keep a mask.

    Returns:
        List of dicts, one per prompt:
            - mask: binary mask (H, W) as uint8 numpy array
            - score: confidence float
            - bbox: (x, y, w, h) bounding box
            - prompt: the original prompt string
    """
    model, cfg = _ensure_model()

    from demo.inference_utils import (
        build_inference_inputs,
        prepare_image_inputs,
    )

    # Save image to temp file for the existing pipeline, then adapt
    import tempfile
    import cv2
    import os

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
        cv2.imwrite(tmp_path, image)

    try:
        _, sam_tensor, beit_tensor, height, width = prepare_image_inputs(
            tmp_path, cfg.INPUT.FORMAT
        )
    finally:
        os.unlink(tmp_path)

    prompts = [p.strip() for p in prompts]
    category_ids = list(range(len(prompts)))

    inputs = build_inference_inputs(
        sam_tensor, beit_tensor, height, width, prompts, category_ids
    )

    with torch.no_grad():
        outputs = model(inputs)[0]

    grounding_masks = outputs.get("grounding_mask")
    grounding_scores = outputs.get("grounding_scores")

    if grounding_masks is None:
        raise RuntimeError("Model returned no referring expression masks.")

    if torch.is_tensor(grounding_masks):
        mask_array = grounding_masks.detach().cpu().numpy()
    else:
        mask_array = np.asarray(grounding_masks)

    if torch.is_tensor(grounding_scores):
        scores = grounding_scores.detach().cpu().tolist()
    else:
        scores = list(grounding_scores)

    results = []
    for idx, (mask, score) in enumerate(zip(mask_array, scores)):
        if mask.ndim == 3:
            mask = mask.squeeze(0)

        binary_mask = (mask >= 0.5).astype(np.uint8)

        # Compute bbox from mask
        ys, xs = np.where(binary_mask > 0)
        if len(xs) > 0:
            x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())
            bbox = (x, y, w, h)
        else:
            bbox = (0, 0, 0, 0)

        results.append({
            "mask": binary_mask,
            "score": float(score),
            "bbox": bbox,
            "prompt": prompts[idx] if idx < len(prompts) else "",
        })

    return results
