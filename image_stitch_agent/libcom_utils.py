"""
Optional image composition utilities — simOPA scoring + color harmonization.

Integrates with BCMI Lab's libcom (pip install libcom) when available.
Includes built-in Reinhard color transfer as a zero-dependency fallback.
"""

import logging
import types
import numpy as np
import cv2
import torch

logger = logging.getLogger(__name__)

# Lazy-loaded model instances
_OPA_MODEL = None
_has_libcom = None  # None=not checked, True=available, False=unavailable


def check_libcom() -> bool:
    """Fast check — is libcom usable? Safe to call at UI build time."""
    global _has_libcom
    if _has_libcom is not None:
        return _has_libcom
    _has_libcom = _try_import_libcom()
    if not _has_libcom:
        logger.info("libcom unavailable — using built-in color transfer only.")
    return _has_libcom


def _try_import_libcom() -> bool:
    """Try to import libcom, handling known broken submodules gracefully."""
    import sys

    # Pre-mock submodules known to have missing dependencies in libcom 0.1.7
    # This prevents cascade failures when libcom/__init__.py imports everything
    broken = [
        'libcom.image_harmonization',
        'libcom.image_harmonization.image_harmonization',
        'libcom.painterly_image_harmonization',
        'libcom.painterly_image_harmonization.painterly_image_harmonization',
        'libcom.shadow_generation',
        'libcom.shadow_generation.shadow_generation',
        'libcom.reflection_generation',
        'libcom.reflection_generation.reflection_generation',
        'libcom.kontext_blending_harmonization',
        'libcom.kontext_blending_harmonization.kontext_blending_harmonization',
        'libcom.os_insert',
        'libcom.os_insert.os_insert',
        'libcom.inharmonious_region_localization',
        'libcom.inharmonious_region_localization.inharmonious_region_localization',
        'libcom.fopa_heat_map',
        'libcom.fopa_heat_map.fopa_heat_map',
        'libcom.fos_score',
        'libcom.fos_score.fos_score',
        'libcom.harmony_score',
        'libcom.harmony_score.harmony_score',
    ]
    for name in broken:
        sys.modules[name] = types.ModuleType(name)

    try:
        import libcom  # noqa: F401
        # Verify the module we need is actually importable
        from libcom.opa_score import OPAScoreModel  # noqa: F401
        logger.info("libcom loaded successfully (OPAScoreModel available).")
        return True
    except Exception as e:
        logger.info("libcom import failed: %s", e)
        return False


def _ensure_opa_model():
    """Lazy-load simOPA model using our own opa_scorer (CPU supported)."""
    global _OPA_MODEL
    if _OPA_MODEL is not None:
        return True
    try:
        from .opa_scorer import _get_scorer
        _OPA_MODEL = _get_scorer('cpu')
        return _OPA_MODEL is not None
    except Exception as e:
        logger.warning("Failed to load simOPA: %s", e)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Reinhard Color Transfer — pure algorithm, zero dependencies beyond numpy+opencv
# Based on: Reinhard et al., "Color Transfer between Images", IEEE CG&A 2001
# ═══════════════════════════════════════════════════════════════════════════════

def reinhard_color_transfer(
    foreground: np.ndarray,
    background: np.ndarray,
    fg_mask: np.ndarray,
) -> np.ndarray:
    """Adjust foreground colors to match background statistics.

    Computes mean/std of both images in Lab color space, then linearly
    transforms the foreground's Lab channels to match the background's.

    Args:
        foreground: RGB image (H, W, 3) uint8.
        background: RGB image (H, W, 3) uint8.
        fg_mask: Binary mask of the foreground (same size as foreground).

    Returns:
        Color-adjusted RGB foreground image.
    """
    fg = foreground.astype(np.float32)
    bg = background.astype(np.float32)
    mask = (fg_mask > 0).astype(np.uint8)

    # Convert to Lab
    fg_lab = cv2.cvtColor((fg / 255.0).astype(np.float32), cv2.COLOR_RGB2Lab)
    bg_lab = cv2.cvtColor((bg / 255.0).astype(np.float32), cv2.COLOR_RGB2Lab)

    # Statistics of the foreground object (masked)
    mask_3ch = np.stack([mask] * 3, axis=-1)
    fg_pixels = fg_lab[mask_3ch > 0].reshape(-1, 3)
    if len(fg_pixels) < 2:
        return foreground

    fg_mean = fg_pixels.mean(axis=0)
    fg_std = fg_pixels.std(axis=0) + 1e-6

    # Statistics of the entire background
    bg_pixels = bg_lab.reshape(-1, 3)
    bg_mean = bg_pixels.mean(axis=0)
    bg_std = bg_pixels.std(axis=0) + 1e-6

    # Linear transform: map fg mean/std to bg mean/std
    ratio = bg_std / fg_std
    transformed_lab = (fg_lab - fg_mean) * ratio + bg_mean

    # Clamp Lab values: L ∈ [0,100], a ∈ [-128,127], b ∈ [-128,127]
    transformed_lab[:, :, 0] = np.clip(transformed_lab[:, :, 0], 0, 100)
    transformed_lab[:, :, 1] = np.clip(transformed_lab[:, :, 1], -128, 127)
    transformed_lab[:, :, 2] = np.clip(transformed_lab[:, :, 2], -128, 127)

    # Convert back to RGB
    transformed_rgb = cv2.cvtColor(transformed_lab.astype(np.float32), cv2.COLOR_Lab2RGB)
    transformed_rgb = (transformed_rgb * 255).clip(0, 255).astype(np.uint8)

    # Only update foreground pixels
    result = foreground.copy()
    result[mask_3ch > 0] = transformed_rgb[mask_3ch > 0]

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def score_composition(
    composite: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    """Score composited image naturalness using simOPA (our own CPU build).

    Args:
        composite: RGB composited image (H, W, 3) uint8.
        mask: Binary mask (H, W) uint8 of the inserted region.

    Returns:
        Float score (higher = more natural), or None if model unavailable.
    """
    from .opa_scorer import score_composition as _sc
    return _sc(composite, mask)


def harmonize_libcom(
    background: np.ndarray,
    foreground: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray | None:
    """Adjust foreground colors to match background.

    Uses Reinhard color transfer (always available, no extra deps).
    """
    try:
        return reinhard_color_transfer(foreground, background, mask)
    except Exception as e:
        logger.warning("Color transfer failed: %s", e)
        return None
