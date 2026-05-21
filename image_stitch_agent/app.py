"""
OpenSplice — SAM 3 segmentation + pose-aware image stitching.
Gradio web UI.

Usage:
    python -m image_stitch_agent.app
"""

import logging
import cv2
import gradio as gr
import numpy as np

from .config import OUTPUT_DIR
from .segmenter import Segmenter, mask_overlay, mask_contour, _ensure_model
from .image_gen_client import generate_image, load_source_image, harmonize_image
from .stitcher import poisson_blend, resize_and_crop_to_mask, crop_to_object

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Preload SAM 3 once at startup
print("Preloading SAM 3 model...")
_ensure_model()
print("SAM 3 ready.")

# Session state
_state = {
    "image": None,
    "segmenter": None,
    "masks": [],
    "selected": 0,
    "replacement": None,
    "box_start": None,
    "point_clicks": [],
}


def _new_image(img):
    """Reset state for a new image."""
    _state["image"] = img
    _state["segmenter"] = Segmenter(img)
    _state["masks"] = []
    _state["selected"] = 0
    _state["replacement"] = None
    _state["box_start"] = None
    _state["point_clicks"] = []
    return img, f"Image: {img.shape[1]}x{img.shape[0]}. Ready."


# ---------------------------------------------------------------------------
# Image input
# ---------------------------------------------------------------------------

def on_input(img):
    if img is None:
        return None, "No image."
    # Skip if same image data — prevents spurious state reset when Gradio
    # fires duplicate events. Uses array_equal (not `is`) because Gradio may
    # create new numpy array objects for the same underlying image.
    cur = _state.get("image")
    if cur is not None and cur.shape == img.shape and np.array_equal(cur, img):
        return img, f"Image already loaded: {img.shape[1]}x{img.shape[0]}. Ready."
    vis, msg = _new_image(img)
    return vis, msg


# ---------------------------------------------------------------------------
# Text prompt
# ---------------------------------------------------------------------------

def on_text(prompt):
    seg = _state.get("segmenter")
    img = _state.get("image")
    if seg is None:
        return _state.get("image"), "Upload/capture an image first."

    masks = seg.segment_by_text(prompt.strip())
    _state["masks"] = masks
    _state["selected"] = 0

    if not masks:
        return img, f"No objects found for '{prompt}'."

    colors = [(0, 180, 0), (180, 0, 0), (0, 0, 180), (0, 180, 180), (180, 180, 0), (180, 0, 180)]
    vis = img.copy()
    for i, m in enumerate(masks):
        c = colors[i % len(colors)]
        vis = mask_overlay(vis, m["mask"], c)
        vis = mask_contour(vis, m["mask"], (0, 255, 0), 2)

    # Highlight selected mask
    vis = _highlight_selected(vis)

    info = f"TEXT '{prompt}': {len(masks)} result(s)  |  Selected: [0]"
    for i, m in enumerate(masks[:6]):
        info += f"\n  [{i}] IoU={m['score']:.3f}  area={m['mask'].sum()}px"
    return vis, info


def _highlight_selected(vis):
    """Draw a thicker highlight on the selected mask."""
    masks = _state.get("masks", [])
    idx = _state.get("selected", 0)
    if masks and idx < len(masks):
        vis = mask_contour(vis, masks[idx]["mask"], (0, 255, 0), 4)
    return vis


# ---------------------------------------------------------------------------
# Interactive clicks (point & box modes)
# ---------------------------------------------------------------------------

def on_result_click(vis, mode, evt: gr.SelectData):
    """Handle click on result image — point or box mode."""
    seg = _state.get("segmenter")
    img = _state.get("image")
    if seg is None or img is None:
        return vis, "Upload/capture an image first."

    x, y = evt.index[0], evt.index[1]

    if mode == "box":
        return _handle_box_click(vis, x, y)
    else:
        return _handle_point_click(vis, x, y)


def _handle_box_click(vis, x, y):
    box_start = _state.get("box_start")
    img = _state["image"]
    seg = _state["segmenter"]

    if box_start is None:
        _state["box_start"] = (x, y)
        cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.circle(vis, (x, y), 10, (0, 0, 255), 2)
        return vis, f"Box corner 1: ({x},{y}). Click opposite corner..."

    # Second corner — run box segmentation
    x1, y1 = box_start
    x1c, x2c = min(x1, x), max(x1, x)
    y1c, y2c = min(y1, y), max(y1, y)
    _state["box_start"] = None

    masks = seg.segment_by_box([x1c, y1c, x2c, y2c])
    _state["masks"] = masks
    _state["selected"] = 0

    if not masks:
        return img.copy(), f"Nothing in box [{x1c},{y1c},{x2c},{y2c}]."

    vis2 = img.copy()
    info = f"BOX [{x1c},{y1c},{x2c},{y2c}]: {len(masks)} result(s)  |  Selected: [0]"
    for i, m in enumerate(masks):
        vis2 = mask_overlay(vis2, m["mask"], (0, 120, 200))
        vis2 = mask_contour(vis2, m["mask"], (255, 200, 0), 2)
        info += f"\n  [{i}] IoU={m['score']:.3f}  area={m['mask'].sum()}px"
    cv2.rectangle(vis2, (x1c, y1c), (x2c, y2c), (255, 0, 0), 2)
    vis2 = _highlight_selected(vis2)
    return vis2, info


def _handle_point_click(vis, x, y):
    _state["point_clicks"].append((x, y, 1))
    n = len(_state["point_clicks"])

    cv2.circle(vis, (x, y), 8, (0, 255, 0), -1)
    cv2.circle(vis, (x, y), 10, (255, 255, 255), 2)
    cv2.putText(vis, str(n), (x + 14, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis, f"Point #{n} at ({x},{y}) foreground. {n} point(s) pending."


def on_point_segment():
    seg = _state.get("segmenter")
    img = _state.get("image")
    clicks = _state.get("point_clicks", [])

    if seg is None:
        return img, "Upload/capture an image first."
    if not clicks:
        return img, "Click on image to add points first."

    points = [[x, y] for x, y, _ in clicks]
    labels = [l for _, _, l in clicks]

    masks = seg.segment_by_point(points, labels)
    _state["masks"] = masks
    _state["selected"] = 0
    _state["point_clicks"] = []

    if not masks:
        return img.copy(), "No object found at those points."

    vis = img.copy()
    info = f"POINTS ({len(points)}): {len(masks)} result(s)  |  Selected: [0]"
    colors = [(200, 0, 0), (0, 200, 0), (0, 0, 200)]
    for i, m in enumerate(masks):
        c = colors[i % len(colors)]
        vis = mask_overlay(vis, m["mask"], c)
        vis = mask_contour(vis, m["mask"], (0, 255, 0), 2)
        info += f"\n  [{i}] IoU={m['score']:.3f}  area={m['mask'].sum()}px"
    vis = _highlight_selected(vis)
    return vis, info


def on_clear():
    _state["box_start"] = None
    _state["point_clicks"] = []
    return _state.get("image"), "Cleared. Ready."


def on_mode_change(mode):
    _state["box_start"] = None
    _state["point_clicks"] = []
    return _state.get("image"), f"Mode: {mode.upper()}"


# ---------------------------------------------------------------------------
# Mask selection
# ---------------------------------------------------------------------------

def on_select(idx):
    img = _state.get("image")
    masks = _state.get("masks")
    if img is None or not masks:
        return img, "No masks."

    idx = max(0, min(int(idx), len(masks) - 1))
    _state["selected"] = idx
    m = masks[idx]

    vis = img.copy()
    vis = mask_overlay(vis, m["mask"], (0, 255, 0))
    vis = _highlight_selected(vis)

    return vis, f"Selected [{idx}]: IoU={m['score']:.3f}  bbox={m['bbox']}  area={m['mask'].sum()}px"


# ---------------------------------------------------------------------------
# Replacement
# ---------------------------------------------------------------------------

def on_repl_upload(f):
    if f is None:
        return None, "No file."
    try:
        # Gradio 6 returns a file object (path in .name), Gradio 5 returns a str
        path = f.name if hasattr(f, 'name') else str(f)
        img = load_source_image(path)
        _state["replacement"] = img
        print(f"[DEBUG] Replacement loaded: {img.shape[1]}x{img.shape[0]} from {path}")
        return img, f"Replacement: {img.shape[1]}x{img.shape[0]}"
    except Exception as e:
        print(f"[DEBUG] Replacement upload failed: {e}")
        return None, f"Error: {e}"


def on_generate(prompt):
    if not prompt.strip():
        return None, "Enter a prompt."
    try:
        img = generate_image(prompt.strip())
        _state["replacement"] = img
        return img, f"Generated: {img.shape[1]}x{img.shape[0]}"
    except Exception as e:
        return None, f"Generation failed: {e}"


# ---------------------------------------------------------------------------
# Stitching
# ---------------------------------------------------------------------------

def _do_fast_stitch():
    """Core fast stitch: Poisson-blend replacement into image. Returns (rgb_image, error)."""
    img = _state.get("image")
    masks = _state.get("masks")
    repl = _state.get("replacement")
    idx = _state.get("selected", 0)

    if img is None:
        return None, "No image. Upload an image first."
    if not masks:
        return None, "No masks. Segment an object first."
    if repl is None:
        return None, "No replacement. Upload or generate one first."
    if idx >= len(masks):
        return None, f"Invalid mask index {idx} (have {len(masks)} masks)."

    m = masks[idx]
    mask = m["mask"]

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, "Empty mask."
    w, h = int(xs.max() - xs.min()), int(ys.max() - ys.min())

    # Remove background from replacement — extract just the main object
    repl_obj = crop_to_object(repl)
    print(f"[DEBUG] crop_to_object: {repl.shape} -> {repl_obj.shape}")

    repl_rs = resize_and_crop_to_mask(repl_obj, mask)
    adapted = cv2.resize(repl_rs, (w, h), interpolation=cv2.INTER_CUBIC)
    result = poisson_blend(img, adapted, mask)
    return result, None


def on_fast_stitch():
    print("[DEBUG] on_fast_stitch called")
    try:
        result, err = _do_fast_stitch()
        if result is None:
            return None, err
        out = str(OUTPUT_DIR / "last_fast.png")
        cv2.imwrite(out, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"[DEBUG] FastStitch done: {out}")
        return result, f"Done: {out}"
    except Exception as e:
        logger.exception("FastStitch failed")
        print(f"[DEBUG] FastStitch ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {e}"


def on_stitch():
    """Pose-Adapt & Stitch: fast stitch first, then AI harmonizes the composite."""
    print("[DEBUG] on_stitch called")
    try:
        # Step 1: Fast stitch (Poisson blend)
        stitched, err = _do_fast_stitch()
        if stitched is None:
            return None, err
        print(f"[DEBUG] Fast stitch done, sending to AI harmonization...")

        # Step 2: AI harmonization — fix seams, lighting, scale mismatch
        harmonize_prompt = (
            "This is a composited image where an object was pasted into the scene. "
            "The pasted object looks out of place — fix it to look completely natural. "
            "Adjust: (1) lighting and shadows to match the surrounding scene, "
            "(2) edge blending so there are no visible seams or hard borders, "
            "(3) color tone and white balance to match the background, "
            "(4) scale and perspective if the object looks too big/small. "
            "Keep the background and everything outside the pasted region IDENTICAL. "
            "Only modify the pasted object and its immediate edges. "
            "The result should look like a single original photograph."
        )
        result = harmonize_image(stitched, harmonize_prompt)
        out = str(OUTPUT_DIR / "last_result.png")
        cv2.imwrite(out, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"[DEBUG] Stitch done: {out}")
        return result, f"Done: {out}"
    except Exception as e:
        logger.exception("Stitch failed")
        print(f"[DEBUG] Stitch ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {e}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_CSS = """
.section { border: 1px solid #555; border-radius: 8px; padding: 12px; margin: 8px 0; }
.section h3 { margin-top: 0; }
"""

def build_ui():
    with gr.Blocks(title="OpenSplice — AI Image Stitching") as app:
        gr.Markdown("# OpenSplice — AI Image Stitching")
        gr.Markdown("Segment objects with SAM 3 (text / box / point), then replace & stitch.")

        with gr.Row():
            # Left: images
            with gr.Column(scale=3):
                input_img = gr.Image(
                    label="1. Input Image",
                    type="numpy",
                    sources=["upload", "webcam"],
                    height=340,
                )
                result_img = gr.Image(
                    label="2. Segmentation (click to add points or draw box)",
                    type="numpy",
                    height=340,
                    interactive=True,
                )
                repl_img = gr.Image(
                    label="3. Replacement Preview",
                    type="numpy",
                    height=200,
                )
                final_img = gr.Image(
                    label="4. Stitched Result",
                    type="numpy",
                    height=300,
                )
                status = gr.Textbox(label="Status", value="Upload an image to begin.", lines=3, interactive=False)

            # Right: controls
            with gr.Column(scale=2):
                # --- Click mode ---
                gr.HTML('<div class="section"><h3>Interaction Mode</h3>')
                mode = gr.Radio(choices=["box", "point"], value="box", label="Mode")
                gr.Markdown("**Box**: click two corners to draw a box.  **Point**: click to add foreground points.")
                gr.HTML("</div>")

                # --- Text ---
                gr.HTML('<div class="section"><h3>Text Prompt</h3>')
                text_prompt = gr.Textbox(placeholder="e.g. 'a person', 'the red car', '白色的杯子'", label="Describe object")
                text_btn = gr.Button("Segment by Text", variant="primary")
                gr.HTML("</div>")

                # --- Point ---
                gr.HTML('<div class="section"><h3>Point Controls</h3>')
                with gr.Row():
                    pt_btn = gr.Button("Segment Points", variant="primary")
                    clear_btn = gr.Button("Clear Points", variant="secondary")
                gr.HTML("</div>")

                # --- Mask selection ---
                gr.HTML('<div class="section"><h3>Mask Selection</h3>')
                mask_idx = gr.Number(label="Selected Mask #", value=0, precision=0, minimum=0)
                select_btn = gr.Button("Highlight This Mask", variant="secondary")
                gr.HTML("</div>")

                # --- Replacement ---
                gr.HTML('<div class="section"><h3>Replacement Image</h3>')
                repl_file = gr.File(label="Upload replacement", file_types=["image"])
                gen_prompt = gr.Textbox(placeholder="Or describe to generate: 'a golden retriever'", label="Generate (AI)")
                gen_btn = gr.Button("Generate Replacement", variant="secondary")
                gr.HTML("</div>")

                # --- Stitch ---
                gr.HTML('<div class="section"><h3>Stitch</h3>')
                with gr.Row():
                    stitch_btn = gr.Button("Pose-Adapt & Stitch (AI)", variant="primary")
                    fast_btn = gr.Button("Fast Stitch (no API)", variant="secondary")
                gr.HTML("</div>")

        # --- Wiring ---
        input_img.upload(on_input, [input_img], [result_img, status])
        input_img.change(on_input, [input_img], [result_img, status])

        text_btn.click(on_text, [text_prompt], [result_img, status])

        result_img.select(on_result_click, [result_img, mode], [result_img, status])

        mode.change(on_mode_change, [mode], [result_img, status])

        pt_btn.click(on_point_segment, [], [result_img, status])
        clear_btn.click(on_clear, [], [result_img, status])

        select_btn.click(on_select, [mask_idx], [result_img, status])

        repl_file.change(on_repl_upload, [repl_file], [repl_img, status])
        gen_btn.click(on_generate, [gen_prompt], [repl_img, status])

        stitch_btn.click(on_stitch, [], [final_img, status])
        fast_btn.click(on_fast_stitch, [], [final_img, status])

    return app


def main():
    ui = build_ui()
    ui.queue().launch(server_name="127.0.0.1", server_port=7860, css=_CSS)


if __name__ == "__main__":
    main()
