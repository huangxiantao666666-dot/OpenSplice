"""
Quick test for SAM 3 segmentation — text, box, and point prompts.

Usage:
    python test_segmentation.py [image_path]

Default: outputs/classroom.png
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from image_stitch_agent.segmenter import Segmenter, mask_overlay, mask_contour
from image_stitch_agent.config import OUTPUT_DIR


def show_result(img, masks, title=""):
    """Print mask info and return a composite overlay."""
    print(f"\n{title}")
    if not masks:
        print("  No masks found.")
        return img.copy()

    colors = [
        (0, 0, 200), (0, 200, 0), (200, 0, 0),
        (0, 200, 200), (200, 200, 0), (200, 0, 200),
    ]
    result = img.copy()
    for i, m in enumerate(masks):
        color = colors[i % len(colors)]
        score = m.get("score", 0)
        bbox = m.get("bbox", (0, 0, 0, 0))
        area = m["mask"].sum()
        print(f"  [{i}] score={score:.3f}, bbox=({bbox[0]},{bbox[1]},{bbox[2]}x{bbox[3]}), area={area}px")
        alpha = 0.35
        result[m["mask"] > 0] = (result[m["mask"] > 0] * (1 - alpha) +
                                 np.array(color, dtype=np.uint8) * alpha).astype(np.uint8)
        result = mask_contour(result, m["mask"], (0, 255, 0), 2)
    return result


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/classroom.png"

    print(f"Loading: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot read {image_path}")
        sys.exit(1)
    print(f"Image: {img.shape[1]}x{img.shape[0]}")

    print("\nLoading SAM 3 (first call takes ~60s on CPU)...")
    t0 = time.time()
    seg = Segmenter(img)
    print(f"Model ready in {time.time() - t0:.1f}s")

    # ---- 1. Text prompt ----
    print("\n" + "=" * 50)
    print("1. TEXT PROMPT: 'a person'")
    print("=" * 50)
    t0 = time.time()
    masks_text = seg.segment_by_text("a person")
    print(f"Took {time.time() - t0:.1f}s, found {len(masks_text)} mask(s)")
    vis_text = show_result(img, masks_text, "Results:")
    out = OUTPUT_DIR / "seg_test_text.png"
    cv2.imwrite(str(out), vis_text)
    print(f"Saved: {out}")

    # ---- 2. Box prompt ----
    print("\n" + "=" * 50)
    print("2. BOX PROMPT: manual bbox around center-left region")
    print("=" * 50)
    # Draw a box in the center-left area of the image
    h, w = img.shape[:2]
    box = [w // 4, h // 4, w // 2, h // 2]  # [x1, y1, x2, y2]
    print(f"  Box: {box}")
    t0 = time.time()
    masks_box = seg.segment_by_box(box)
    print(f"Took {time.time() - t0:.1f}s, found {len(masks_box)} mask(s)")
    vis_box = show_result(img, masks_box, "Results:")
    # Draw the prompt box on the result
    x1, y1, x2, y2 = box
    cv2.rectangle(vis_box, (x1, y1), (x2, y2), (255, 0, 0), 2)
    out = OUTPUT_DIR / "seg_test_box.png"
    cv2.imwrite(str(out), vis_box)
    print(f"Saved: {out}")

    # ---- 3. Point prompt ----
    print("\n" + "=" * 50)
    print("3. POINT PROMPT: click at image center")
    print("=" * 50)
    cx, cy = w // 2, h // 2
    print(f"  Point: ({cx}, {cy}) as foreground")
    t0 = time.time()
    masks_point = seg.segment_by_point(points=[[cx, cy]], labels=[1])
    print(f"Took {time.time() - t0:.1f}s, found {len(masks_point)} mask(s)")
    vis_point = show_result(img, masks_point, "Results:")
    # Draw the point on the result
    cv2.circle(vis_point, (cx, cy), 10, (0, 0, 255), -1)
    cv2.circle(vis_point, (cx, cy), 12, (255, 255, 255), 2)
    out = OUTPUT_DIR / "seg_test_point.png"
    cv2.imwrite(str(out), vis_point)
    print(f"Saved: {out}")

    print("\n" + "=" * 50)
    print("DONE — check outputs/ for results:")
    print(f"  {OUTPUT_DIR / 'seg_test_text.png'}")
    print(f"  {OUTPUT_DIR / 'seg_test_box.png'}")
    print(f"  {OUTPUT_DIR / 'seg_test_point.png'}")


if __name__ == "__main__":
    main()
