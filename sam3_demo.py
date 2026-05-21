"""
SAM 3 Interactive Segmentation Demo — text, box, and point prompts.

Usage:
    python sam3_demo.py
    Then open http://127.0.0.1:7860

Click/drag on the image for box or point prompts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import gradio as gr
import numpy as np

from image_stitch_agent.segmenter import (
    Segmenter, mask_overlay, mask_contour, _ensure_model,
)

# Preload model at startup
print("Loading SAM 3 (this takes ~60s on CPU)...")
_ensure_model()
print("SAM 3 ready!")

# State
_seg = None
_image = None
_box_start = None     # (x, y) — first corner of pending box
_point_clicks = []    # [(x, y, label), ...]


def _new_segmenter(img):
    """Replace the segmenter for a new image."""
    global _seg, _image, _box_start, _point_clicks
    _image = img
    _seg = Segmenter(img)
    _box_start = None
    _point_clicks = []
    h, w = img.shape[:2]
    return f"New image: {w}x{h}. Ready."


# ---------------------------------------------------------------------------
# Image input (upload or webcam)
# ---------------------------------------------------------------------------

def on_input(img):
    if img is None:
        return None, "No image."
    # Skip if same image data — prevents spurious state reset when Gradio
    # fires duplicate events. Uses array_equal (not `is`) because Gradio may
    # create new numpy array objects for the same underlying image.
    if _image is not None and _image.shape == img.shape and np.array_equal(_image, img):
        return img, f"Image already loaded: {img.shape[1]}x{img.shape[0]}. Ready."
    msg = _new_segmenter(img)
    return img, msg


# ---------------------------------------------------------------------------
# TEXT prompt
# ---------------------------------------------------------------------------

def on_text(prompt):
    if _seg is None:
        return _image, "Upload/capture an image first."
    if not prompt.strip():
        return _image, "Enter a text prompt."

    masks = _seg.segment_by_text(prompt.strip())
    if not masks:
        return _image, f"Nothing found for '{prompt}'."

    colors = [
        (0, 180, 0), (180, 0, 0), (0, 0, 180),
        (0, 180, 180), (180, 180, 0), (180, 0, 180),
    ]
    vis = _image.copy()
    info = f"TEXT '{prompt}': {len(masks)} result(s)"
    for i, m in enumerate(masks):
        c = colors[i % len(colors)]
        vis = mask_overlay(vis, m["mask"], c)
        vis = mask_contour(vis, m["mask"], (0, 255, 0), 2)
        info += f"\n  [{i}] IoU={m['score']:.3f}  area={m['mask'].sum()}px  bbox={m['bbox']}"
    return vis, info


# ---------------------------------------------------------------------------
# Interactive clicks — supports both POINT and BOX modes
# ---------------------------------------------------------------------------

def on_result_click(vis, mode, evt: gr.SelectData):
    """Handle clicks on the result image — point or box mode."""
    global _box_start, _point_clicks

    if _seg is None or _image is None:
        return vis, "Upload/capture an image first."

    x, y = evt.index[0], evt.index[1]

    if mode == "box":
        if _box_start is None:
            # First corner — show marker and wait
            _box_start = (x, y)
            cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.circle(vis, (x, y), 10, (0, 0, 255), 2)
            return vis, f"Box corner 1: ({x},{y}). Click opposite corner..."

        # Second corner — run box segmentation
        x1, y1 = _box_start
        x1c, x2c = min(x1, x), max(x1, x)
        y1c, y2c = min(y1, y), max(y1, y)
        _box_start = None

        masks = _seg.segment_by_box([x1c, y1c, x2c, y2c])
        if not masks:
            return _image.copy(), f"Nothing found in box [{x1c},{y1c},{x2c},{y2c}]."

        vis2 = _image.copy()
        info = f"BOX [{x1c},{y1c},{x2c},{y2c}]: {len(masks)} result(s)"
        for i, m in enumerate(masks):
            vis2 = mask_overlay(vis2, m["mask"], (0, 120, 200))
            vis2 = mask_contour(vis2, m["mask"], (255, 200, 0), 2)
            info += f"\n  [{i}] IoU={m['score']:.3f}  area={m['mask'].sum()}px  bbox={m['bbox']}"
        cv2.rectangle(vis2, (x1c, y1c), (x2c, y2c), (255, 0, 0), 2)
        return vis2, info

    else:  # point mode
        _point_clicks.append((x, y, 1))
        # Draw the point
        cv2.circle(vis, (x, y), 8, (0, 255, 0), -1)
        cv2.circle(vis, (x, y), 10, (255, 255, 255), 2)
        # Draw point number
        cv2.putText(vis, str(len(_point_clicks)), (x + 14, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info = f"Point #{len(_point_clicks)} at ({x},{y}) — foreground"
        info += f"\n{len(_point_clicks)} point(s) pending. Click 'Segment Points' or click more points."
        return vis, info


def on_point_segment():
    global _point_clicks
    if _seg is None:
        return _image, "Upload/capture an image first."
    if not _point_clicks:
        return _image, "Click on the image first to add points."

    points = [[x, y] for x, y, _ in _point_clicks]
    labels = [lbl for _, _, lbl in _point_clicks]

    masks = _seg.segment_by_point(points, labels)
    _point_clicks = []

    if not masks:
        return _image.copy(), "No object found at those points."

    vis = _image.copy()
    info = f"POINTS ({len(points)}): {len(masks)} result(s)"
    colors = [(200, 0, 0), (0, 200, 0), (0, 0, 200)]
    for i, m in enumerate(masks):
        c = colors[i % len(colors)]
        vis = mask_overlay(vis, m["mask"], c)
        vis = mask_contour(vis, m["mask"], (0, 255, 0), 2)
        info += f"\n  [{i}] IoU={m['score']:.3f}  area={m['mask'].sum()}px  bbox={m['bbox']}"
    return vis, info


def on_clear():
    global _box_start, _point_clicks
    _box_start = None
    _point_clicks = []
    return _image, "Cleared. Ready."


def on_mode_change(mode):
    global _box_start, _point_clicks
    _box_start = None
    _point_clicks = []
    return _image, f"Switched to {mode.upper()} mode."


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_CSS = """
.prompt-section { border: 1px solid #555; border-radius: 8px; padding: 12px; margin: 8px 0; }
.prompt-section h3 { margin-top: 0; }
"""

def build_ui():
    with gr.Blocks(title="SAM 3 Segmentation Demo") as app:
        gr.Markdown("# SAM 3 — Interactive Segmentation Demo")

        with gr.Row():
            # Left: input + result images
            with gr.Column(scale=3):
                input_img = gr.Image(
                    label="Input (upload or webcam)",
                    type="numpy",
                    sources=["upload", "webcam"],
                    height=380,
                )
                result_img = gr.Image(
                    label="Result — click/drag to add prompts",
                    type="numpy",
                    height=380,
                    interactive=True,
                )
                status = gr.Textbox(label="Status / Results", lines=3, interactive=False)

            # Right: controls
            with gr.Column(scale=1):
                # Interactive mode selector
                gr.Markdown("### Interaction Mode")
                mode = gr.Radio(
                    choices=["point", "box"],
                    value="box",
                    label="Click mode",
                )
                gr.Markdown("**Box**: click two corners to draw a box.  **Point**: click to add foreground points.")

                # Text prompt
                gr.HTML('<div class="prompt-section"><h3>Text Prompt</h3>')
                text_prompt = gr.Textbox(
                    placeholder="e.g. 'a person', 'the dog', '穿红色衣服的人'",
                    label="Describe object",
                )
                text_btn = gr.Button("Segment by Text", variant="primary")
                gr.HTML("</div>")

                # Point controls
                gr.HTML('<div class="prompt-section"><h3>Point Controls</h3>')
                with gr.Row():
                    point_btn = gr.Button("Segment Points", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")
                gr.HTML("</div>")

        # Wiring
        input_img.upload(on_input, [input_img], [result_img, status])
        input_img.change(on_input, [input_img], [result_img, status])

        text_btn.click(on_text, [text_prompt], [result_img, status])

        result_img.select(on_result_click, [result_img, mode], [result_img, status])

        point_btn.click(on_point_segment, [], [result_img, status])
        clear_btn.click(on_clear, [], [result_img, status])
        mode.change(on_mode_change, [mode], [result_img, status])

    return app


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="127.0.0.1", server_port=7860, css=_CSS)
