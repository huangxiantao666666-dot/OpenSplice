"""
Image stitching and blending using Poisson fusion (cv2.seamlessClone).
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def poisson_blend(
    background: np.ndarray,
    foreground: np.ndarray,
    mask: np.ndarray,
    center: Optional[tuple[int, int]] = None,
    blend_mode: int = cv2.NORMAL_CLONE,
) -> np.ndarray:
    """
    Blend foreground onto background using Poisson image editing.

    Args:
        background: BGR image (H, W, 3) uint8 – the original image.
        foreground: BGR image (H, W, 3) uint8 – the generated patch.
        mask: Binary mask (H, W) uint8 defining the blend region on background.
        center: (x, y) center of the blend region. Defaults to mask centroid.
        blend_mode: cv2.NORMAL_CLONE, cv2.MIXED_CLONE, or cv2.MONOCHROME_TRANSFER.

    Returns:
        Blended BGR image (H, W, 3) uint8.
    """
    assert background.shape[:2] == mask.shape[:2], (
        f"Background shape {background.shape[:2]} != mask shape {mask.shape[:2]}"
    )

    # Resize foreground to fit the mask bounding box
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        logger.warning("Empty mask, returning background unchanged.")
        return background.copy()

    x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())

    if w < 2 or h < 2:
        logger.warning("Mask region too small (%dx%d), returning background unchanged.", w, h)
        return background.copy()

    fg_resized = cv2.resize(foreground, (w, h), interpolation=cv2.INTER_CUBIC)

    if center is None:
        center = (x + w // 2, y + h // 2)

    # Place resized foreground into a full-size canvas at the mask bbox
    fg_full = background.copy()
    fg_full[y:y + h, x:x + w] = fg_resized

    # Use rectangular bbox mask (not the precise mask contour) because the
    # generated object shape rarely matches the original segmented object shape.
    mask_full = np.zeros(background.shape[:2], dtype=np.uint8)
    mask_full[y:y + h, x:x + w] = 255

    try:
        result = cv2.seamlessClone(
            fg_full, background, mask_full, center, blend_mode
        )
    except cv2.error as e:
        logger.warning("seamlessClone failed: %s. Falling back to simple alpha blend.", e)
        result = _alpha_blend(background, fg_full, mask_full)

    return result


def _alpha_blend(
    background: np.ndarray,
    foreground: np.ndarray,
    mask: np.ndarray,
    feather_radius: int = 15,
) -> np.ndarray:
    """Simple alpha blending with Gaussian feathering on mask edges."""
    mask_f = mask.astype(np.float32) / 255.0

    # Gaussian blur the mask for smooth edges
    mask_blurred = cv2.GaussianBlur(mask_f, (feather_radius * 2 + 1, feather_radius * 2 + 1), 0)

    mask_3ch = np.stack([mask_blurred] * 3, axis=-1)
    result = (foreground * mask_3ch + background * (1 - mask_3ch)).astype(np.uint8)

    # Only blend where mask > 0, keep original elsewhere
    keep_mask = (mask > 0).astype(np.uint8)
    keep_3ch = np.stack([keep_mask] * 3, axis=-1)
    result = (result * keep_3ch + background * (1 - keep_3ch)).astype(np.uint8)
    return result


def resize_and_crop_to_mask(
    source_img: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    """
    Resize and optionally crop a source image to fit a target mask's aspect ratio.
    Uses the mask's bounding box aspect ratio.

    Returns the resized source image ready for blending.
    """
    ys, xs = np.where(target_mask > 0)
    if len(xs) == 0:
        return source_img

    w = int(xs.max() - xs.min())
    h = int(ys.max() - ys.min())

    # Resize source to match mask bbox aspect ratio, filling with center crop
    src_h, src_w = source_img.shape[:2]
    target_aspect = w / h if h > 0 else 1.0
    src_aspect = src_w / src_h if src_h > 0 else 1.0

    if src_aspect > target_aspect:
        # Source is wider — crop width
        new_w = int(src_h * target_aspect)
        offset = (src_w - new_w) // 2
        cropped = source_img[:, offset:offset + new_w]
    else:
        # Source is taller — crop height
        new_h = int(src_w / target_aspect)
        offset = (src_h - new_h) // 2
        cropped = source_img[offset:offset + new_h, :]

    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)
