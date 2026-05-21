"""
Image stitching and blending using Poisson fusion (cv2.seamlessClone).
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np


def crop_to_object(image: np.ndarray) -> np.ndarray:
    """Crop replacement image to its main foreground object, removing background.

    Uses Canny edge detection + contour finding. Works best on AI-generated
    images where the subject is centered on a plain background.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    edges = cv2.Canny(gray, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    # Keep contours near image center (the main subject)
    cx, cy = w // 2, h // 2
    valid = []
    for c in contours:
        if cv2.contourArea(c) < 50:
            continue
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx_c = int(M["m10"] / M["m00"])
            cy_c = int(M["m01"] / M["m00"])
            if np.sqrt((cx_c - cx) ** 2 + (cy_c - cy) ** 2) < max(w, h) * 0.4:
                valid.append(c)

    if not valid:
        valid = [c for c in contours if cv2.contourArea(c) > 100]
    if not valid:
        return image

    all_pts = np.vstack(valid)
    x, y, bw, bh = cv2.boundingRect(all_pts)

    # Tight crop with small padding
    pad = 8
    x = max(0, x - pad)
    y = max(0, y - pad)
    bw = min(w - x, bw + 2 * pad)
    bh = min(h - y, bh + 2 * pad)

    return image[y:y + bh, x:x + bw]

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

    # Place resized foreground onto a full-size canvas at the mask bbox
    fg_full = background.copy()
    fg_full[y:y + h, x:x + w] = fg_resized

    # Use the precise mask contour so only the object region is blended.
    # smooth the mask edges slightly for a cleaner seam.
    # SAM masks are binary (0/1), so threshold at 0.5 after blur
    mask_blurred = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
    mask_uint8 = (mask_blurred > 0.5).astype(np.uint8) * 255

    try:
        result = cv2.seamlessClone(
            fg_full, background, mask_uint8, center, blend_mode
        )
    except cv2.error as e:
        logger.warning("seamlessClone failed: %s. Falling back to simple alpha blend.", e)
        result = _alpha_blend(background, fg_full, mask_uint8)

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


def make_context_image(
    original: np.ndarray,
    mask: np.ndarray,
    padding: int = 20,
) -> np.ndarray:
    """
    Create a context image for the pose-adaptation model.

    The result shows the original image within the mask's bounding box
    (expanded by padding), with everything outside the bbox set to black.
    This gives the image editing model visual context about the target
    pose, lighting, and surrounding area.

    Returns:
        BGR image same size as original, with non-bbox area blacked out.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros_like(original)

    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(original.shape[1], int(xs.max()) + padding)
    y2 = min(original.shape[0], int(ys.max()) + padding)

    ctx = np.zeros_like(original)
    ctx[y1:y2, x1:x2] = original[y1:y2, x1:x2]
    return ctx


def pose_aware_blend(
    background: np.ndarray,
    mask: np.ndarray,
    replacement: np.ndarray,
    context_image: np.ndarray,
) -> np.ndarray:
    """
    Blend a replacement object into the background using pose adaptation.

    1. Calls pose_adapt_image() to make the replacement match the target pose.
    2. Aligns the adapted result with the mask's bbox center.
    3. Cuts out the object using the precise mask (not a rectangle).
    4. Feathers the mask edges for smooth transition.
    5. Composites onto the background.

    The mask is used as-is — mask holes (occluded areas) are naturally
    preserved because the blend only touches mask==1 pixels.

    Args:
        background: BGR original image.
        mask: Binary mask (H, W) for the target region.
        replacement: BGR image of the replacement object (user-provided or generated).
        context_image: BGR image showing the target pose/region (from make_context_image).
        feather_radius: Gaussian blur radius for mask edge smoothing.

    Returns:
        Blended BGR image.
    """
    from .image_gen_client import pose_adapt_image

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return background.copy()

    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max() - xs.min()), int(ys.max() - ys.min())
    cx, cy = x + w // 2, y + h // 2

    if w < 4 or h < 4:
        logger.warning("Mask too small (%dx%d), returning background unchanged.", w, h)
        return background.copy()

    # --- Step 1: Pose-adapt the replacement to match the target region ---
    edit_prompt = (
        "You are given two images. Image 1 shows the original scene with a blacked-out "
        "background — only the target object region is visible (showing its POSE, "
        "POSITION, SIZE, and surrounding CONTEXT). Image 2 is a replacement object "
        "that should be placed into image 1's target position. "
        "Generate an image of the EXACT SAME object/animal/person from image 2 "
        "(same breed, color, texture, identity), but posed, scaled, and oriented "
        "to match the object shown in image 1. Do NOT change the identity or "
        "appearance of the image 2 subject — only adjust its pose, size, and angle. "
        "The output should be a single object on a plain background, ready for "
        "compositing."
    )

    adapted = pose_adapt_image(context_image, replacement, edit_prompt)

    # --- Step 2: Resize adapted image to mask bbox ---
    adapted = cv2.resize(adapted, (w, h), interpolation=cv2.INTER_CUBIC)

    # --- Step 3: Poisson blend with MIXED_CLONE ---
    return poisson_blend(background, adapted, mask, center=(cx, cy))


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
