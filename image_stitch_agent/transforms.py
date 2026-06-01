"""
Geometric transforms for interactive foreground placement.

Rotation, scaling, overlay rendering, and placement-aware Poisson blending.
All images are RGB numpy arrays (uint8, HWC). Masks are binary uint8 (values 0 or 1).
"""

import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)


def apply_transform(
    image: np.ndarray,
    mask: np.ndarray,
    rotation_deg: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply rotation and scale to both image and mask.

    The output canvas auto-expands so no part of the transformed image
    is clipped (uses the full rotated bounding box).

    Args:
        image: RGB image (H, W, 3) uint8.
        mask: Binary mask (H, W) uint8, values 0 or 1.
        rotation_deg: Counter-clockwise rotation in degrees.
        scale: Scale multiplier (> 0, 1.0 = original size).

    Returns:
        (transformed_image, transformed_mask) — both uint8, same spatial size.
    """
    h, w = image.shape[:2]

    if rotation_deg == 0 and abs(scale - 1.0) < 0.001:
        return image.copy(), mask.copy()

    # Rotation matrix around image center
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), rotation_deg, scale)

    # Compute new canvas size so rotated image is not clipped
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    # Adjust translation so the rotated image stays centered in the new canvas
    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy

    transformed_img = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    # Scale mask to 0-255 so INTER_LINEAR interpolation produces meaningful values
    mask_255 = (mask * 255).astype(np.uint8)
    transformed_mask = cv2.warpAffine(
        mask_255, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # Re-binarize after interpolation
    transformed_mask = (transformed_mask >= 128).astype(np.uint8)

    return transformed_img, transformed_mask


def render_overlay(
    background: np.ndarray,
    foreground: np.ndarray,
    fg_mask: np.ndarray,
    center_x: int,
    center_y: int,
    rotation_deg: float,
    scale: float,
) -> np.ndarray:
    """Alpha-composite a transformed foreground onto a copy of the background.

    The foreground is rotated+scaled, then placed so its center lands at
    (center_x, center_y) on the background. Only mask>0 pixels are composited.

    Args:
        background: RGB image (H, W, 3) uint8.
        foreground: RGB image of the object to place.
        fg_mask: Binary mask of the foreground object (same size as foreground).
        center_x, center_y: Where to place the center of the transformed foreground.
        rotation_deg: Rotation in degrees (CCW).
        scale: Scale multiplier.

    Returns:
        RGB image (same size as background) with the foreground composited on top.
    """
    bg_h, bg_w = background.shape[:2]
    result = background.copy()

    # Apply rotation and scale
    fg_t, mask_t = apply_transform(foreground, fg_mask, rotation_deg, scale)

    fg_h, fg_w = fg_t.shape[:2]

    # Place such that foreground center aligns with (center_x, center_y)
    x1 = center_x - fg_w // 2
    y1 = center_y - fg_h // 2

    # Compute valid region (clip to background bounds)
    src_x1 = max(0, -x1)
    src_y1 = max(0, -y1)
    src_x2 = min(fg_w, bg_w - x1)
    src_y2 = min(fg_h, bg_h - y1)

    dst_x1 = max(0, x1)
    dst_y1 = max(0, y1)
    dst_x2 = min(bg_w, x1 + fg_w)
    dst_y2 = min(bg_h, y1 + fg_h)

    # If no overlap, return background unchanged
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return result

    # Alpha composite where mask > 0
    roi = result[dst_y1:dst_y2, dst_x1:dst_x2]
    fg_roi = fg_t[src_y1:src_y2, src_x1:src_x2]
    m_roi = mask_t[src_y1:src_y2, src_x1:src_x2]

    # 3-channel mask for compositing
    m_3ch = np.stack([m_roi] * 3, axis=-1).astype(np.float32)
    # Blend: foreground where mask=1, background where mask=0
    # Smooth edges: use the mask value (0 or 1 after binarization, but we can
    # feather slightly with a small Gaussian blur on the mask before compositing)
    blended = fg_roi * m_3ch + roi * (1.0 - m_3ch)
    result[dst_y1:dst_y2, dst_x1:dst_x2] = blended.astype(np.uint8)

    return result


def place_and_blend(
    background: np.ndarray,
    foreground: np.ndarray,
    fg_mask: np.ndarray,
    center_x: int,
    center_y: int,
    rotation_deg: float,
    scale: float,
    blend_mode: int = cv2.NORMAL_CLONE,
) -> np.ndarray:
    """Poisson-blend a transformed foreground onto the background.

    Applies rotation and scale, creates full-size foreground+mask canvases
    at the user-chosen position, then calls cv2.seamlessClone.

    Args:
        background: RGB image (H, W, 3) uint8.
        foreground: RGB image of the object to place.
        fg_mask: Binary mask of the foreground object (same size as foreground).
        center_x, center_y: Where to place the center of the foreground.
        rotation_deg: Rotation in degrees (CCW).
        scale: Scale multiplier.
        blend_mode: cv2.NORMAL_CLONE, MIXED_CLONE, or MONOCHROME_TRANSFER.

    Returns:
        Blended RGB image (same size as background).
    """
    bg_h, bg_w = background.shape[:2]

    # Apply rotation and scale
    fg_t, mask_t = apply_transform(foreground, fg_mask, rotation_deg, scale)

    fg_h, fg_w = fg_t.shape[:2]

    # --- Check for valid region ---
    x1 = center_x - fg_w // 2
    y1 = center_y - fg_h // 2

    src_x1 = max(0, -x1)
    src_y1 = max(0, -y1)
    src_x2 = min(fg_w, bg_w - x1)
    src_y2 = min(fg_h, bg_h - y1)

    if src_x2 <= src_x1 or src_y2 <= src_y1:
        logger.warning("Foreground placed entirely outside background bounds.")
        return background.copy()

    # --- Create full-size foreground canvas ---
    fg_full = np.zeros_like(background)
    dst_x1 = max(0, x1)
    dst_y1 = max(0, y1)
    dst_x2 = min(bg_w, x1 + fg_w)
    dst_y2 = min(bg_h, y1 + fg_h)

    fg_full[dst_y1:dst_y2, dst_x1:dst_x2] = \
        fg_t[src_y1:src_y2, src_x1:src_x2]

    # --- Create full-size mask (uint8, 0-255) ---
    mask_full = np.zeros((bg_h, bg_w), dtype=np.uint8)
    mask_full[dst_y1:dst_y2, dst_x1:dst_x2] = \
        (mask_t[src_y1:src_y2, src_x1:src_x2] * 255).astype(np.uint8)

    # --- Smooth mask edges ---
    mask_blurred = cv2.GaussianBlur(mask_full.astype(np.float32), (5, 5), 0)
    mask_uint8 = (mask_blurred > 128).astype(np.uint8) * 255

    if mask_uint8.sum() == 0:
        logger.warning("Empty mask after transform. Returning background unchanged.")
        return background.copy()

    # --- Poisson blend ---
    center = (center_x, center_y)
    try:
        result = cv2.seamlessClone(
            fg_full, background, mask_uint8, center, blend_mode
        )
    except cv2.error as e:
        logger.warning("seamlessClone failed: %s. Falling back to simple blend.", e)
        # Fallback: alpha composite using the smoothed mask
        mask_3ch = (mask_uint8.astype(np.float32) / 255.0)
        mask_3ch = np.stack([mask_3ch] * 3, axis=-1)
        result = (fg_full * mask_3ch + background * (1 - mask_3ch)).astype(np.uint8)

    return result


def alpha_place(
    background: np.ndarray,
    foreground: np.ndarray,
    fg_mask: np.ndarray,
    center_x: int,
    center_y: int,
    rotation_deg: float,
    scale: float,
) -> np.ndarray:
    """Simple alpha blend: foreground where mask>0, background elsewhere.

    No gradient blending — just a direct pixel copy inside the mask region.
    Fastest option, works as a baseline for comparison.
    """
    bg_h, bg_w = background.shape[:2]
    fg_t, mask_t = apply_transform(foreground, fg_mask, rotation_deg, scale)
    fg_h, fg_w = fg_t.shape[:2]

    x1 = center_x - fg_w // 2
    y1 = center_y - fg_h // 2

    src_x1 = max(0, -x1)
    src_y1 = max(0, -y1)
    src_x2 = min(fg_w, bg_w - x1)
    src_y2 = min(fg_h, bg_h - y1)

    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return background.copy()

    result = background.copy()
    dst_x1 = max(0, x1)
    dst_y1 = max(0, y1)
    dst_x2 = min(bg_w, x1 + fg_w)
    dst_y2 = min(bg_h, y1 + fg_h)

    roi = result[dst_y1:dst_y2, dst_x1:dst_x2]
    fg_roi = fg_t[src_y1:src_y2, src_x1:src_x2]
    m_roi = mask_t[src_y1:src_y2, src_x1:src_x2]

    # Simple copy: foreground where mask>0
    m_3ch = (m_roi > 0).astype(np.float32)
    m_3ch = np.stack([m_3ch] * 3, axis=-1)
    roi[:] = (fg_roi * m_3ch + roi * (1 - m_3ch)).astype(np.uint8)

    return result
