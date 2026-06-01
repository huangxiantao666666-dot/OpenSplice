"""
OpenSplice — SAM 3 segmentation + interactive object placement + image stitching.
Gradio web UI with two tabs.

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
from .transforms import render_overlay, place_and_blend, alpha_place
from .libcom_utils import score_composition as _score_comp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Preload SAM 3 once at startup
print("Preloading SAM 3 model...")
_ensure_model()
print("SAM 3 ready.")

# ─── State ────────────────────────────────────────────────────────────────────

# Tab 1 — Interactive Placement
_s1 = {
    "bg_image": None,      # np.ndarray, RGB
    "fg_image": None,      # np.ndarray, RGB
    "fg_mask": None,       # np.ndarray, binary uint8
    "place_x": None,       # int, foreground center X on background
    "place_y": None,       # int, foreground center Y on background
    "rotation": 0,         # float, degrees
    "scale": 1.0,          # float, multiplier
    "blend_result": None,  # np.ndarray, last blend output
}

# Tab 2 — SAM3 Stitch (preserved from original)
_s2 = {
    "image": None,
    "segmenter": None,
    "masks": [],
    "selected": 0,
    "replacement": None,
    "box_start": None,
    "point_clicks": [],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Interactive Placement
# ═══════════════════════════════════════════════════════════════════════════════

def _auto_segment_foreground(fg: np.ndarray) -> np.ndarray:
    """Auto-segment foreground using SAM3 center point. Returns binary mask."""
    h, w = fg.shape[:2]
    try:
        seg = Segmenter(fg)
        masks = seg.segment_by_point([[w // 2, h // 2]], [1])
        if masks:
            # Pick mask with largest area
            best = max(masks, key=lambda m: m["mask"].sum())
            print(f"[DEBUG] Auto-segment: {len(masks)} candidates, selected area={best['mask'].sum()}px")
            return best["mask"]
    except Exception as e:
        print(f"[DEBUG] Auto-segment failed: {e}")
    # Fallback: all-ones mask
    return np.ones((h, w), dtype=np.uint8)


def _tab1_render_preview():
    """Render foreground overlay on background at current position/transform."""
    bg = _s1["bg_image"]
    fg = _s1["fg_image"]
    fm = _s1["fg_mask"]

    if bg is None:
        return None, "Load a background image first."

    if fg is None:
        # Show background only
        return bg, "Load a foreground image. Click on the image to place it."

    # Default position = background center if not set
    px = _s1.get("place_x")
    py = _s1.get("place_y")
    if px is None or py is None:
        px, py = bg.shape[1] // 2, bg.shape[0] // 2
        _s1["place_x"], _s1["place_y"] = px, py

    overlay = render_overlay(bg, fg, fm, px, py, _s1["rotation"], _s1["scale"])
    info = (
        f"Position: ({px}, {py})  |  "
        f"Rotation: {_s1['rotation']}°  |  "
        f"Scale: {_s1['scale']:.2f}×  |  "
        f"Foreground: {fg.shape[1]}×{fg.shape[0]}"
    )
    return overlay, info


# ─── Tab 1 Handlers ───────────────────────────────────────────────────────────

def tab1_load_bg(file_obj):
    """Upload background image."""
    if file_obj is None:
        return None, None, "No file selected."
    try:
        path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
        img = load_source_image(path)
        _s1["bg_image"] = img
        _s1["place_x"] = img.shape[1] // 2
        _s1["place_y"] = img.shape[0] // 2
        _s1["blend_result"] = None
        print(f"[DEBUG] Tab1 BG loaded: {img.shape[1]}×{img.shape[0]}")
        preview, status = _tab1_render_preview()
        return img, preview, status
    except Exception as e:
        print(f"[DEBUG] Tab1 BG load failed: {e}")
        return None, None, f"Error: {e}"


def tab1_load_fg(file_obj):
    """Upload foreground image + auto-segment."""
    if file_obj is None:
        return None, None, "No file selected."
    try:
        path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
        img = load_source_image(path)
        print(f"[DEBUG] Tab1 FG loaded: {img.shape[1]}×{img.shape[0]}, auto-segmenting...")
        mask = _auto_segment_foreground(img)
        _s1["fg_image"] = img
        _s1["fg_mask"] = mask
        _s1["blend_result"] = None
        preview, status = _tab1_render_preview()
        return img, preview, status
    except Exception as e:
        print(f"[DEBUG] Tab1 FG load failed: {e}")
        return None, None, f"Error: {e}"


def tab1_generate_fg(prompt: str):
    """AI-generate foreground image + auto-segment."""
    if not prompt.strip():
        return None, None, "Enter a prompt."
    try:
        img = generate_image(prompt.strip())
        print(f"[DEBUG] Tab1 FG generated: {img.shape[1]}×{img.shape[0]}, auto-segmenting...")
        mask = _auto_segment_foreground(img)
        _s1["fg_image"] = img
        _s1["fg_mask"] = mask
        _s1["blend_result"] = None
        preview, status = _tab1_render_preview()
        return img, preview, status
    except Exception as e:
        print(f"[DEBUG] Tab1 FG generate failed: {e}")
        return None, None, f"Generation failed: {e}"


def tab1_click_preview(evt: gr.SelectData):
    """Handle click on live_preview — set position."""
    x, y = evt.index[0], evt.index[1]
    _s1["place_x"] = int(x)
    _s1["place_y"] = int(y)
    preview, status = _tab1_render_preview()
    return preview, int(x), int(y), status


def tab1_set_position(x, y):
    """Handle manual position entry."""
    if x is not None and y is not None:
        _s1["place_x"] = int(x)
        _s1["place_y"] = int(y)
    preview, status = _tab1_render_preview()
    return preview, status


def tab1_on_rotation(val):
    """Rotation slider change."""
    _s1["rotation"] = float(val)
    preview, status = _tab1_render_preview()
    return preview, status


def tab1_on_scale(val):
    """Scale slider change."""
    _s1["scale"] = float(val)
    preview, status = _tab1_render_preview()
    return preview, status


def tab1_fast_blend():
    """Poisson blend foreground at current position/transform."""
    bg = _s1["bg_image"]
    fg = _s1["fg_image"]
    fm = _s1["fg_mask"]
    px = _s1["place_x"]
    py = _s1["place_y"]

    if bg is None:
        return None, "No background."
    if fg is None:
        return None, "No foreground."
    if px is None:
        return None, "Click on the preview to set position."

    try:
        result = place_and_blend(
            bg, fg, fm, px, py,
            _s1["rotation"], _s1["scale"],
        )
        _s1["blend_result"] = result
        out = str(OUTPUT_DIR / "last_placement_fast.png")
        cv2.imwrite(out, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"[DEBUG] Tab1 FastBlend done: {out}")
        return result, f"Done: {out}"
    except Exception as e:
        print(f"[DEBUG] Tab1 FastBlend ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {e}"


def tab1_alpha_blend():
    """Simple alpha blend: foreground in mask, background elsewhere."""
    bg = _s1["bg_image"]
    fg = _s1["fg_image"]
    fm = _s1["fg_mask"]
    px = _s1["place_x"]
    py = _s1["place_y"]

    if bg is None: return None, "No background."
    if fg is None: return None, "No foreground."
    if px is None: return None, "Click on preview to set position."

    try:
        result = alpha_place(bg, fg, fm, px, py, _s1["rotation"], _s1["scale"])
        _s1["blend_result"] = result
        out = str(OUTPUT_DIR / "last_placement_alpha.png")
        cv2.imwrite(out, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        return result, f"Done: {out}"
    except Exception as e:
        return None, f"Error: {e}"


def tab1_ai_harmonize():
    """Fast blend then AI harmonize."""
    bg = _s1["bg_image"]
    fg = _s1["fg_image"]

    if bg is None: return None, "No background."
    if fg is None: return None, "No foreground."

    # Run fast blend first if needed
    if _s1["blend_result"] is None:
        result, err = tab1_fast_blend()
        if result is None:
            return None, err

    try:
        harmonized = harmonize_image(
            _s1["blend_result"],
            "Fix this composited image. Adjust lighting, shadows, edges, color "
            "tone, and perspective of the pasted object to blend seamlessly. "
            "Keep the background identical. Make it look like one original photo."
        )
        _s1["blend_result"] = harmonized
        out = str(OUTPUT_DIR / "last_placement_harmonized.png")
        cv2.imwrite(out, cv2.cvtColor(harmonized, cv2.COLOR_RGB2BGR))
        print(f"[DEBUG] Tab1 AI Harmonize done: {out}")
        return harmonized, f"Done: {out}"
    except Exception as e:
        print(f"[DEBUG] Tab1 AI Harmonize ERROR: {e}")
        return _s1["blend_result"], f"Harmonization failed: {e}"


def tab1_libcom_score():
    """Score the blend result using simOPA."""
    result = _s1.get("blend_result")
    fg = _s1.get("fg_image")
    if result is None or fg is None:
        return "No blend result to score."
    score = _score_comp(result, _s1["fg_mask"])
    if score is None:
        return "simOPA model not loaded. Check checkpoints/simopa.pth"
    # Interpret the score
    if score >= 0.8:
        level = "(excellent)"
    elif score >= 0.5:
        level = "(good)"
    elif score >= 0.2:
        level = "(fair)"
    elif score >= 0.01:
        level = "(poor)"
    else:
        level = "(very poor)"
    return f"Score: {score:.6f}  {level}"


def tab1_color_transfer():
    """Reinhard color transfer foreground → match background colors, then blend."""
    bg = _s1["bg_image"]
    fg = _s1["fg_image"]
    fm = _s1["fg_mask"]
    if bg is None: return None, "No background."
    if fg is None: return None, "No foreground."
    if _s1["place_x"] is None: return None, "Click on preview to set position first."

    try:
        from .libcom_utils import reinhard_color_transfer
        hfg = reinhard_color_transfer(fg, bg, fm)
        result = place_and_blend(
            bg, hfg, fm, _s1["place_x"], _s1["place_y"],
            _s1["rotation"], _s1["scale"],
        )
        _s1["blend_result"] = result
        out = str(OUTPUT_DIR / "last_placement_colortransfer.png")
        cv2.imwrite(out, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        return result, f"Done (color transfer): {out}"
    except Exception as e:
        return None, f"Color transfer failed: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — SAM3 Stitch (preserved from original, handlers renamed tab2_*)
# ═══════════════════════════════════════════════════════════════════════════════

def tab2_new_image(img):
    _s2["image"] = img
    _s2["segmenter"] = Segmenter(img)
    _s2["masks"] = []
    _s2["selected"] = 0
    _s2["replacement"] = None
    _s2["box_start"] = None
    _s2["point_clicks"] = []
    return img, f"Image: {img.shape[1]}x{img.shape[0]}. Ready."


def tab2_on_input(img):
    if img is None:
        return None, "No image."
    cur = _s2.get("image")
    if cur is not None and cur.shape == img.shape and np.array_equal(cur, img):
        return img, f"Image already loaded: {img.shape[1]}x{img.shape[0]}. Ready."
    vis, msg = tab2_new_image(img)
    return vis, msg


def tab2_highlight_selected(vis):
    masks = _s2.get("masks", [])
    idx = _s2.get("selected", 0)
    if masks and idx < len(masks):
        vis = mask_contour(vis, masks[idx]["mask"], (0, 255, 0), 4)
    return vis


def tab2_on_text(prompt):
    seg = _s2.get("segmenter")
    img = _s2.get("image")
    if seg is None:
        return _s2.get("image"), "Upload/capture an image first."
    masks = seg.segment_by_text(prompt.strip())
    _s2["masks"] = masks
    _s2["selected"] = 0
    if not masks:
        return img, f"No objects found for '{prompt}'."
    colors = [(0, 180, 0), (180, 0, 0), (0, 0, 180), (0, 180, 180), (180, 180, 0), (180, 0, 180)]
    vis = img.copy()
    for i, m in enumerate(masks):
        c = colors[i % len(colors)]
        vis = mask_overlay(vis, m["mask"], c)
        vis = mask_contour(vis, m["mask"], (0, 255, 0), 2)
    vis = tab2_highlight_selected(vis)
    info = f"TEXT '{prompt}': {len(masks)} result(s)  |  Selected: [0]"
    for i, m in enumerate(masks[:6]):
        info += f"\n  [{i}] IoU={m['score']:.3f}  area={m['mask'].sum()}px"
    return vis, info


def tab2_on_result_click(vis, mode, evt: gr.SelectData):
    seg = _s2.get("segmenter")
    img = _s2.get("image")
    if seg is None or img is None:
        return vis, "Upload/capture an image first."
    x, y = evt.index[0], evt.index[1]
    if mode == "box":
        return tab2_handle_box_click(vis, x, y)
    else:
        return tab2_handle_point_click(vis, x, y)


def tab2_handle_box_click(vis, x, y):
    box_start = _s2.get("box_start")
    img = _s2["image"]
    seg = _s2["segmenter"]
    if box_start is None:
        _s2["box_start"] = (x, y)
        cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.circle(vis, (x, y), 10, (0, 0, 255), 2)
        return vis, f"Box corner 1: ({x},{y}). Click opposite corner..."
    x1, y1 = box_start
    x1c, x2c = min(x1, x), max(x1, x)
    y1c, y2c = min(y1, y), max(y1, y)
    _s2["box_start"] = None
    masks = seg.segment_by_box([x1c, y1c, x2c, y2c])
    _s2["masks"] = masks
    _s2["selected"] = 0
    if not masks:
        return img.copy(), f"Nothing in box [{x1c},{y1c},{x2c},{y2c}]."
    vis2 = img.copy()
    info = f"BOX [{x1c},{y1c},{x2c},{y2c}]: {len(masks)} result(s)  |  Selected: [0]"
    for i, m in enumerate(masks):
        vis2 = mask_overlay(vis2, m["mask"], (0, 120, 200))
        vis2 = mask_contour(vis2, m["mask"], (255, 200, 0), 2)
        info += f"\n  [{i}] IoU={m['score']:.3f}  area={m['mask'].sum()}px"
    cv2.rectangle(vis2, (x1c, y1c), (x2c, y2c), (255, 0, 0), 2)
    vis2 = tab2_highlight_selected(vis2)
    return vis2, info


def tab2_handle_point_click(vis, x, y):
    _s2["point_clicks"].append((x, y, 1))
    n = len(_s2["point_clicks"])
    cv2.circle(vis, (x, y), 8, (0, 255, 0), -1)
    cv2.circle(vis, (x, y), 10, (255, 255, 255), 2)
    cv2.putText(vis, str(n), (x + 14, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis, f"Point #{n} at ({x},{y}) foreground. {n} point(s) pending."


def tab2_on_point_segment():
    seg = _s2.get("segmenter")
    img = _s2.get("image")
    clicks = _s2.get("point_clicks", [])
    if seg is None: return img, "Upload/capture an image first."
    if not clicks: return img, "Click on image to add points first."
    points = [[x, y] for x, y, _ in clicks]
    labels = [l for _, _, l in clicks]
    masks = seg.segment_by_point(points, labels)
    _s2["masks"] = masks
    _s2["selected"] = 0
    _s2["point_clicks"] = []
    if not masks: return img.copy(), "No object found at those points."
    vis = img.copy()
    info = f"POINTS ({len(points)}): {len(masks)} result(s)  |  Selected: [0]"
    colors = [(200, 0, 0), (0, 200, 0), (0, 0, 200)]
    for i, m in enumerate(masks):
        c = colors[i % len(colors)]
        vis = mask_overlay(vis, m["mask"], c)
        vis = mask_contour(vis, m["mask"], (0, 255, 0), 2)
        info += f"\n  [{i}] IoU={m['score']:.3f}  area={m['mask'].sum()}px"
    vis = tab2_highlight_selected(vis)
    return vis, info


def tab2_on_clear():
    _s2["box_start"] = None
    _s2["point_clicks"] = []
    return _s2.get("image"), "Cleared. Ready."


def tab2_on_mode_change(mode):
    _s2["box_start"] = None
    _s2["point_clicks"] = []
    return _s2.get("image"), f"Mode: {mode.upper()}"


def tab2_on_select(idx):
    img = _s2.get("image")
    masks = _s2.get("masks")
    if img is None or not masks: return img, "No masks."
    idx = max(0, min(int(idx), len(masks) - 1))
    _s2["selected"] = idx
    m = masks[idx]
    vis = img.copy()
    vis = mask_overlay(vis, m["mask"], (0, 255, 0))
    vis = tab2_highlight_selected(vis)
    return vis, f"Selected [{idx}]: IoU={m['score']:.3f}  bbox={m['bbox']}  area={m['mask'].sum()}px"


def tab2_on_repl_upload(f):
    if f is None: return None, "No file."
    try:
        path = f.name if hasattr(f, 'name') else str(f)
        img = load_source_image(path)
        _s2["replacement"] = img
        print(f"[DEBUG] Tab2 Replacement loaded: {img.shape[1]}x{img.shape[0]}")
        return img, f"Replacement: {img.shape[1]}x{img.shape[0]}"
    except Exception as e:
        return None, f"Error: {e}"


def tab2_on_generate(prompt):
    if not prompt.strip(): return None, "Enter a prompt."
    try:
        img = generate_image(prompt.strip())
        _s2["replacement"] = img
        return img, f"Generated: {img.shape[1]}x{img.shape[0]}"
    except Exception as e:
        return None, f"Generation failed: {e}"


def tab2_do_fast_stitch():
    """Core fast stitch for Tab 2."""
    img = _s2.get("image")
    masks = _s2.get("masks")
    repl = _s2.get("replacement")
    idx = _s2.get("selected", 0)
    if img is None: return None, "No image."
    if not masks: return None, "No masks."
    if repl is None: return None, "No replacement."
    if idx >= len(masks): return None, f"Bad mask index {idx}."
    m = masks[idx]
    mask = m["mask"]
    ys, xs = np.where(mask > 0)
    if len(xs) == 0: return None, "Empty mask."
    w, h = int(xs.max() - xs.min()), int(ys.max() - ys.min())
    repl_obj = crop_to_object(repl)
    repl_rs = resize_and_crop_to_mask(repl_obj, mask)
    adapted = cv2.resize(repl_rs, (w, h), interpolation=cv2.INTER_CUBIC)
    result = poisson_blend(img, adapted, mask)
    return result, None


def tab2_on_fast_stitch():
    try:
        result, err = tab2_do_fast_stitch()
        if result is None: return None, err
        out = str(OUTPUT_DIR / "last_fast.png")
        cv2.imwrite(out, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        return result, f"Done: {out}"
    except Exception as e:
        logger.exception("FastStitch failed")
        return None, f"Error: {e}"


def tab2_on_stitch():
    """Fast stitch + AI harmonize."""
    try:
        stitched, err = tab2_do_fast_stitch()
        if stitched is None: return None, err
        harmonized = harmonize_image(
            stitched,
            "Fix this composited image. Adjust lighting, shadows, edges, color "
            "tone, and perspective of the pasted object to blend seamlessly. "
            "Keep the background identical. Make it look like one original photo."
        )
        out = str(OUTPUT_DIR / "last_result.png")
        cv2.imwrite(out, cv2.cvtColor(harmonized, cv2.COLOR_RGB2BGR))
        return harmonized, f"Done: {out}"
    except Exception as e:
        logger.exception("Stitch failed")
        return None, f"Error: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════

_CSS = """
.section { border: 1px solid #555; border-radius: 8px; padding: 12px; margin: 8px 0; }
.section h3 { margin-top: 0; }
.libcom-section { border: 1px solid #4a9; border-radius: 8px; padding: 12px; margin: 8px 0; }
"""


def build_ui():
    with gr.Blocks(title="OpenSplice — AI Image Stitching") as app:
        gr.Markdown("# OpenSplice — AI Image Stitching")

        with gr.Tabs():

            # ── Tab 1: Interactive Placement ──────────────────────────────────
            with gr.Tab("Interactive Placement"):
                gr.Markdown(
                    "Upload a background and foreground. Click on the preview "
                    "to position the foreground, then use the sliders to rotate "
                    "and scale. The AI auto-segments the foreground."
                )

                with gr.Row():
                    # Left column: images
                    with gr.Column(scale=3):
                        bg_img = gr.Image(
                            label="Background", type="numpy", height=200,
                        )
                        fg_img = gr.Image(
                            label="Foreground (auto-segmented)", type="numpy", height=200,
                        )
                        live_preview = gr.Image(
                            label="Preview — click to position the foreground",
                            type="numpy", interactive=True, height=360,
                        )
                        blend_result1 = gr.Image(
                            label="Blend Result", type="numpy", height=300,
                        )
                        status1 = gr.Textbox(
                            label="Status",
                            value="1) Load background. 2) Load/generate foreground. 3) Click to position. 4) Blend.",
                            lines=2, interactive=False,
                        )

                    # Right column: controls
                    with gr.Column(scale=2):
                        gr.HTML('<div class="section"><h3>Load</h3>')
                        upload_bg = gr.File(label="Upload Background", file_types=["image"])
                        upload_fg = gr.File(label="Upload Foreground", file_types=["image"])
                        gen_prompt1 = gr.Textbox(
                            placeholder="Or describe to generate: 'a golden retriever'",
                            label="Generate Foreground (AI)",
                        )
                        gen_btn1 = gr.Button("Generate", variant="secondary")
                        gr.HTML("</div>")

                        gr.HTML('<div class="section"><h3>Position</h3>')
                        with gr.Row():
                            pos_x = gr.Number(label="X", value=0, precision=0)
                            pos_y = gr.Number(label="Y", value=0, precision=0)
                        gr.HTML("</div>")

                        gr.HTML('<div class="section"><h3>Transform</h3>')
                        rotation_slider = gr.Slider(
                            label="Rotation (°)", minimum=-180, maximum=180,
                            value=0, step=1,
                        )
                        scale_slider = gr.Slider(
                            label="Scale", minimum=0.1, maximum=3.0,
                            value=1.0, step=0.05,
                        )
                        gr.HTML("</div>")

                        gr.HTML('<div class="section"><h3>Blend</h3>')
                        with gr.Row():
                            alpha_blend_btn = gr.Button("Alpha Blend", variant="secondary")
                            fast_blend_btn = gr.Button("Fast Blend (Poisson)", variant="primary")
                        ai_harmonize_btn = gr.Button("AI Harmonize", variant="secondary")
                        gr.HTML("</div>")

                        # Color adjustment section
                        gr.HTML('<div class="section"><h3>Color Adjust</h3>')
                        with gr.Row():
                            color_xfer_btn = gr.Button(
                                "Color Transfer (Reinhard)", variant="secondary",
                            )
                        gr.HTML("</div>")

                        # simOPA scoring (built-in, no libcom needed)
                        gr.HTML('<div class="section"><h3>Scoring</h3>')
                        score_btn = gr.Button("Score Naturalness (simOPA)")
                        score_display = gr.Textbox(label="Score", interactive=False)
                        gr.HTML("</div>")

                # Tab 1 Wiring
                upload_bg.change(tab1_load_bg, [upload_bg], [bg_img, live_preview, status1])
                upload_fg.change(tab1_load_fg, [upload_fg], [fg_img, live_preview, status1])
                gen_btn1.click(tab1_generate_fg, [gen_prompt1], [fg_img, live_preview, status1])

                live_preview.select(
                    tab1_click_preview, [], [live_preview, pos_x, pos_y, status1],
                )
                pos_x.change(tab1_set_position, [pos_x, pos_y], [live_preview, status1])
                pos_y.change(tab1_set_position, [pos_x, pos_y], [live_preview, status1])

                rotation_slider.change(tab1_on_rotation, [rotation_slider], [live_preview, status1])
                scale_slider.change(tab1_on_scale, [scale_slider], [live_preview, status1])

                alpha_blend_btn.click(tab1_alpha_blend, [], [blend_result1, status1])
                fast_blend_btn.click(tab1_fast_blend, [], [blend_result1, status1])
                ai_harmonize_btn.click(tab1_ai_harmonize, [], [blend_result1, status1])

                color_xfer_btn.click(tab1_color_transfer, [], [blend_result1, status1])
                score_btn.click(tab1_libcom_score, [], [score_display])

            # ── Tab 2: SAM3 Stitch ────────────────────────────────────────────
            with gr.Tab("SAM3 Stitch"):
                gr.Markdown(
                    "Segment objects with SAM 3 (text / box / point), "
                    "then replace & stitch."
                )

                with gr.Row():
                    with gr.Column(scale=3):
                        input_img = gr.Image(
                            label="1. Input Image", type="numpy",
                            sources=["upload", "webcam"], height=340,
                        )
                        result_img = gr.Image(
                            label="2. Segmentation (click to add points/draw box)",
                            type="numpy", height=340, interactive=True,
                        )
                        repl_img = gr.Image(
                            label="3. Replacement Preview", type="numpy", height=200,
                        )
                        final_img = gr.Image(
                            label="4. Stitched Result", type="numpy", height=300,
                        )
                        status2 = gr.Textbox(
                            label="Status", value="Upload an image to begin.",
                            lines=3, interactive=False,
                        )

                    with gr.Column(scale=2):
                        gr.HTML('<div class="section"><h3>Interaction Mode</h3>')
                        mode = gr.Radio(choices=["box", "point"], value="box", label="Mode")
                        gr.Markdown("**Box**: click two corners. **Point**: click to add points.")
                        gr.HTML("</div>")

                        gr.HTML('<div class="section"><h3>Text Prompt</h3>')
                        text_prompt = gr.Textbox(
                            placeholder="e.g. 'a person', 'the red car', '白色的杯子'",
                            label="Describe object",
                        )
                        text_btn = gr.Button("Segment by Text", variant="primary")
                        gr.HTML("</div>")

                        gr.HTML('<div class="section"><h3>Point Controls</h3>')
                        with gr.Row():
                            pt_btn = gr.Button("Segment Points", variant="primary")
                            clear_btn = gr.Button("Clear Points", variant="secondary")
                        gr.HTML("</div>")

                        gr.HTML('<div class="section"><h3>Mask Selection</h3>')
                        mask_idx = gr.Number(
                            label="Selected Mask #", value=0, precision=0, minimum=0,
                        )
                        select_btn = gr.Button("Highlight This Mask", variant="secondary")
                        gr.HTML("</div>")

                        gr.HTML('<div class="section"><h3>Replacement Image</h3>')
                        repl_file = gr.File(label="Upload replacement", file_types=["image"])
                        gen_prompt2 = gr.Textbox(
                            placeholder="Or describe to generate: 'a golden retriever'",
                            label="Generate (AI)",
                        )
                        gen_btn2 = gr.Button("Generate Replacement", variant="secondary")
                        gr.HTML("</div>")

                        gr.HTML('<div class="section"><h3>Stitch</h3>')
                        with gr.Row():
                            stitch_btn = gr.Button("Pose-Adapt & Stitch (AI)", variant="primary")
                            fast_btn = gr.Button("Fast Stitch (no API)", variant="secondary")
                        gr.HTML("</div>")

                # Tab 2 Wiring
                input_img.upload(tab2_on_input, [input_img], [result_img, status2])
                input_img.change(tab2_on_input, [input_img], [result_img, status2])

                text_btn.click(tab2_on_text, [text_prompt], [result_img, status2])
                result_img.select(
                    tab2_on_result_click, [result_img, mode], [result_img, status2],
                )
                mode.change(tab2_on_mode_change, [mode], [result_img, status2])

                pt_btn.click(tab2_on_point_segment, [], [result_img, status2])
                clear_btn.click(tab2_on_clear, [], [result_img, status2])
                select_btn.click(tab2_on_select, [mask_idx], [result_img, status2])

                repl_file.change(tab2_on_repl_upload, [repl_file], [repl_img, status2])
                gen_btn2.click(tab2_on_generate, [gen_prompt2], [repl_img, status2])

                stitch_btn.click(tab2_on_stitch, [], [final_img, status2])
                fast_btn.click(tab2_on_fast_stitch, [], [final_img, status2])

    return app


def main():
    ui = build_ui()
    ui.queue().launch(server_name="127.0.0.1", server_port=7860, css=_CSS)


if __name__ == "__main__":
    main()
