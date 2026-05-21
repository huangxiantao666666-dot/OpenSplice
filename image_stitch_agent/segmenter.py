"""
SAM 3 wrapper — segmentation via text, point, and box prompts.
Uses Sam3Processor for text grounding + predict_inst for fine masks.
Supports Chinese and English text descriptions.
"""

import logging
from typing import List, Optional

import cv2
import numpy as np
import PIL.Image
import torch

from .config import SAM3_CHECKPOINT, SAM3_DEVICE

if not torch.cuda.is_available():
    torch.set_default_device("cpu")

logger = logging.getLogger(__name__)

_model = None       # Sam3Image model (with inst_interactive_predictor)
_processor = None   # Sam3Processor (for set_image + text grounding)


def _ensure_model():
    """Lazy-load SAM 3 with interactive predictor enabled for fine masks."""
    global _model, _processor
    if _model is not None:
        return _model, _processor

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    logger.info("Loading SAM 3 (device=%s, interactive=True) ...", SAM3_DEVICE)
    model = build_sam3_image_model(
        checkpoint_path=SAM3_CHECKPOINT,
        device=SAM3_DEVICE,
        enable_inst_interactivity=True,  # enables SAM1-style point/box segmentation
    )
    processor = Sam3Processor(model, device=SAM3_DEVICE)
    logger.info("SAM 3 loaded (interactive predictor available).")
    _model = model
    _processor = processor
    return model, processor


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return image[:, :, ::-1].copy()


class Segmenter:
    """Holds inference state for a single image."""

    def __init__(self, image: np.ndarray):
        model, processor = _ensure_model()
        rgb = _bgr_to_rgb(image)
        self._image_shape = image.shape[:2]
        # Sam3Processor.set_image has a bug: uses shape[-2:] for numpy arrays
        # which gives (W, C) for HWC format. Work around by passing PIL Image.
        pil_image = PIL.Image.fromarray(rgb)
        self._state = processor.set_image(pil_image)

    # ------------------------------------------------------------------
    # Text prompt — grounding (detection), then refine with box
    # ------------------------------------------------------------------

    def segment_by_text(self, prompt: str) -> List[dict]:
        """Find objects by text concept. Returns fine masks refined via predict_inst."""
        _processor.set_text_prompt(prompt=prompt, state=self._state)
        # _state now has "boxes" (xyxy pixels) and coarse "masks" from grounding
        return self._refine_from_state(prompt)

    # ------------------------------------------------------------------
    # Point prompt — interactive SAM1-style prediction
    # ------------------------------------------------------------------

    def segment_by_point(
        self,
        points: List[List[float]],
        labels: List[int],
    ) -> List[dict]:
        """Segment with point prompts using the interactive predictor.

        points: [[x, y], ...] in pixel coordinates
        labels: [1, 0, ...]  where 1=foreground, 0=background
        """
        if not points:
            return []

        point_coords = np.array(points, dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int32)

        # Single point without box is ambiguous → get 3 candidates
        multimask = (len(points) == 1 and len(labels) == 1 and labels[0] == 1)

        masks, ious, _ = _model.predict_inst(
            inference_state=self._state,
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask,
        )
        return self._masks_to_dicts(masks, ious, "point")

    # ------------------------------------------------------------------
    # Box prompt — interactive SAM1-style prediction
    # ------------------------------------------------------------------

    def segment_by_box(self, box: List[float]) -> List[dict]:
        """Segment with box prompt using the interactive predictor.

        box: [x1, y1, x2, y2] in pixel coordinates
        """
        masks, ious, _ = _model.predict_inst(
            inference_state=self._state,
            box=np.array(box, dtype=np.float32),
            multimask_output=False,
        )
        return self._masks_to_dicts(masks, ious, "box")

    # ------------------------------------------------------------------
    # Refine grounding masks with predict_inst
    # ------------------------------------------------------------------

    def _refine_from_state(self, prompt: str) -> List[dict]:
        """Take boxes from text grounding, refine each with predict_inst."""
        boxes = self._state.get("boxes")
        scores = self._state.get("scores")

        if boxes is None or len(boxes) == 0:
            return []

        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().tolist()

        results = []
        for i in range(len(boxes)):
            # boxes from grounding are in xyxy pixel format
            b = boxes[i]
            # predict_inst expects [x1, y1, x2, y2] numpy array
            masks, ious, _ = _model.predict_inst(
                inference_state=self._state,
                box=b.astype(np.float32) if isinstance(b, np.ndarray) else np.array(b, dtype=np.float32),
                multimask_output=False,
            )
            # Take best mask (first one since multimask_output=False)
            if masks is not None and len(masks) > 0:
                m = masks[0]
                score = float(ious[0]) if ious is not None and len(ious) > 0 else float(scores[i]) if isinstance(scores, list) else float(scores)

                ys, xs = np.where(m > 0.5)
                if len(xs) < 4:
                    continue
                bbox = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min()))

                results.append({
                    "mask": (m >= 0.5).astype(np.uint8),
                    "score": score,
                    "bbox": bbox,
                    "prompt": prompt,
                })

        return results

    # ------------------------------------------------------------------
    # Convert predict_inst output to mask dicts
    # ------------------------------------------------------------------

    def _masks_to_dicts(self, masks: np.ndarray, ious: np.ndarray | None, prompt: str) -> List[dict]:
        """Convert predict_inst output to list of mask dicts."""
        results = []
        for i in range(len(masks)):
            m = masks[i]
            binary = (m >= 0.5).astype(np.uint8)

            ys, xs = np.where(binary > 0)
            if len(xs) < 4:
                continue
            bbox = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min()))

            score = float(ious[i]) if ious is not None and i < len(ious) else 0.0

            results.append({
                "mask": binary,
                "score": score,
                "bbox": bbox,
                "prompt": prompt,
            })

        return results


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def mask_overlay(image: np.ndarray, mask: np.ndarray, color=(0, 0, 200)) -> np.ndarray:
    overlay = image.copy()
    overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array(color, dtype=np.uint8) * 0.5).astype(np.uint8)
    return overlay


def mask_contour(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), thickness=2) -> np.ndarray:
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = image.copy()
    cv2.drawContours(result, contours, -1, color, thickness)
    return result
