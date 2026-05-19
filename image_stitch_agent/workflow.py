"""
LangGraph workflow for image stitching agent.
"""

import logging
import time
from pathlib import Path
from typing import TypedDict, Optional, Any

import cv2
import numpy as np

from langgraph.graph import StateGraph, END

from .config import OUTPUT_DIR
from .llm_client import decompose_task, vision_check
from .owsam_wrapper import segment
from .image_gen_client import generate_image, load_source_image, harmonize_image
from .stitcher import poisson_blend, resize_and_crop_to_mask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    user_instruction: str
    original_image_path: str
    original_image: Any
    plan: Optional[dict]
    masks: dict
    generated_images: dict
    current_step_idx: int
    result_image: Optional[Any]
    review_result: Optional[dict]
    harmonized: bool
    error: Optional[str]


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _has_error(state: AgentState) -> str:
    return "error" if state.get("error") else "ok"


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def load_image_node(state: AgentState) -> AgentState:
    """Load the original image from disk. Resolves relative paths."""
    img_path = state["original_image_path"]
    path = Path(img_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    logger.info("Loading image: %s", path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        state["error"] = f"Cannot read image: {path}"
        return state
    state["original_image"] = img
    return state


def decompose_node(state: AgentState) -> AgentState:
    """LLM (vision model) decomposes user instruction into a task plan."""
    print("\n" + "=" * 60)
    print("  STEP 1: Task Decomposition (Vision LLM)")
    print("=" * 60)
    try:
        plan = decompose_task(state["user_instruction"], state["original_image"])
        state["plan"] = plan
        state["masks"] = {}
        state["generated_images"] = {}
        state["current_step_idx"] = 0
        state["harmonized"] = False

        steps = plan.get("steps", [])
        print(f"  Plan: {len(steps)} steps")
        for s in steps:
            if s.get("action") == "segment":
                print(f"    [{s['step_id']}] SEGMENT: {s['target_description'][:80]}")
            elif s.get("action") == "generate":
                prompt = s.get("generation_prompt", "") or s.get("source_image_path", "")
                print(f"    [{s['step_id']}] GENERATE: {prompt[:80]}")
        placement = plan.get("final_placement", {})
        print(f"  Blend: step {placement.get('source_step')} -> step {placement.get('paste_region_step')} ({placement.get('blend_mode', 'poisson')})")
    except Exception as e:
        state["error"] = f"Task decomposition failed: {e}"
        logger.exception("Decomposition error")
    return state


def segment_node(state: AgentState) -> AgentState:
    """Run OpenWorldSAM on all segment steps in the plan."""
    print("\n" + "=" * 60)
    print("  STEP 2: Segmentation (OpenWorldSAM)")
    print("=" * 60)
    plan = state["plan"]
    image = state["original_image"]

    if plan is None:
        state["error"] = "No plan to execute (decomposition failed)"
        return state

    segment_steps = [s for s in plan.get("steps", []) if s.get("action") == "segment"]
    if not segment_steps:
        print("  No segment steps to run.")
        return state

    prompts = [s["target_description"] for s in segment_steps]
    print(f"  Running {len(prompts)} segmentation(s)...")

    try:
        results = segment(image, prompts)
        for step, result in zip(segment_steps, results):
            state["masks"][step["step_id"]] = result
            area = result["mask"].sum()
            print(f"    [{step['step_id']}] '{step['target_description'][:60]}'")
            print(f"         mask area={area}px, score={result['score']:.3f}")
    except Exception as e:
        state["error"] = f"Segmentation failed: {e}"
        logger.exception("Segmentation error")

    return state


def generate_node(state: AgentState) -> AgentState:
    """Run image generation for all generate steps."""
    print("\n" + "=" * 60)
    print("  STEP 3: Image Generation (Z-Image-Turbo)")
    print("=" * 60)
    plan = state["plan"]
    if plan is None:
        state["error"] = "No plan to execute"
        return state

    gen_steps = [s for s in plan.get("steps", []) if s.get("action") == "generate"]
    if not gen_steps:
        return state

    for step in gen_steps:
        step_id = step["step_id"]
        source_path = step.get("source_image_path")

        try:
            if source_path:
                print(f"  [{step_id}] Loading source: {source_path}")
                img = load_source_image(source_path)
            else:
                prompt = step.get("generation_prompt", "")
                if not prompt:
                    state["error"] = f"Step {step_id}: no generation_prompt or source_image_path"
                    return state
                print(f"  [{step_id}] Generating image...")
                print(f"       prompt: {prompt[:120]}...")
                img = generate_image(prompt)

            state["generated_images"][step_id] = img
            print(f"       result: {img.shape[1]}x{img.shape[0]}")
        except Exception as e:
            state["error"] = f"Image generation failed at step {step_id}: {e}"
            logger.exception("Generation error")
            return state

    return state


def stitch_node(state: AgentState) -> AgentState:
    """Stitch generated images onto the original using Poisson blending."""
    print("\n" + "=" * 60)
    print("  STEP 4: Stitching (Poisson Blend)")
    print("=" * 60)
    plan = state["plan"]
    if plan is None:
        state["error"] = "No plan to execute"
        return state

    placement = plan.get("final_placement", {})
    image = state["original_image"].copy()

    paste_step_id = placement.get("paste_region_step")
    source_step_id = placement.get("source_step")
    blend_mode_name = placement.get("blend_mode", "poisson")

    blend_map = {
        "poisson": cv2.NORMAL_CLONE,
        "mixed": cv2.MIXED_CLONE,
        "monochrome": cv2.MONOCHROME_TRANSFER,
    }
    blend_mode = blend_map.get(blend_mode_name, cv2.NORMAL_CLONE)

    mask_info = state["masks"].get(paste_step_id)
    if mask_info is None:
        state["error"] = f"Mask for step {paste_step_id} not found. Available: {list(state['masks'].keys())}"
        return state

    gen_img = state["generated_images"].get(source_step_id)
    if gen_img is None:
        state["error"] = f"Generated image for step {source_step_id} not found. Available: {list(state['generated_images'].keys())}"
        return state

    print(f"  Mask bbox: {mask_info['bbox']}")
    print(f"  Blend mode: {blend_mode_name}")

    gen_resized = resize_and_crop_to_mask(gen_img, mask_info["mask"])

    try:
        result = poisson_blend(image, gen_resized, mask_info["mask"], blend_mode=blend_mode)
        state["result_image"] = result
        print(f"  Stitching complete.")
    except Exception as e:
        state["error"] = f"Stitching failed: {e}"
        logger.exception("Stitch error")

    return state


def review_node(state: AgentState) -> AgentState:
    """Visual inspection of the result. Sets up harmonization prompt if needed."""
    print("\n" + "=" * 60)
    print("  STEP 5: Visual Review (Vision LLM)")
    print("=" * 60)
    if state["result_image"] is None:
        state["error"] = "No result image to review"
        return state

    plan = state.get("plan", {})
    gen_steps = [s for s in plan.get("steps", []) if s.get("action") == "generate"]
    gen_prompt = gen_steps[0].get("generation_prompt", "") if gen_steps else ""

    try:
        result = vision_check(
            original_image=state["original_image"],
            modified_image=state["result_image"],
            user_instruction=state["user_instruction"],
            generation_prompt=gen_prompt,
        )
        state["review_result"] = result

        approved = result.get("approved", True)
        score = result.get("score", "?")
        issues = result.get("issues", [])
        feedback = result.get("feedback", "")
        print(f"  Approved: {approved}, Score: {score}/10")
        for issue in issues:
            print(f"    - {issue}")

        if not approved and not state.get("harmonized", False):
            print(f"\n  -> Will harmonize (in-place image edit)")
            print(f"  Edit instruction: {feedback[:150]}")
        elif approved:
            print(f"\n  -> Result accepted!")
    except Exception as e:
        logger.warning("Vision check failed: %s. Approving by default.", e)
        state["review_result"] = {"approved": True, "score": 7, "issues": [], "feedback": ""}
        state["error"] = None

    return state


def harmonize_node(state: AgentState) -> AgentState:
    """Harmonize the stitched image in-place using Qwen Image Edit."""
    print("\n" + "=" * 60)
    print("  STEP 6: Harmonization (Qwen-Image-Edit)")
    print("=" * 60)

    review = state.get("review_result", {})
    edit_prompt = (
        "Fix the visual integration of the edited region in this composite image. "
        "Match the lighting, color temperature, and shadows to the surrounding scene. "
        "Remove edge artifacts, halos, and unnatural transitions. "
        "Make the pasted object look like it naturally belongs in the scene.\n\n"
        f"Specific issues to fix: {review.get('feedback', 'Make the edit look natural and seamless.')}"
    )

    print(f"  Edit prompt: {edit_prompt[:150]}...")

    try:
        harmonized = harmonize_image(state["result_image"], edit_prompt)
        state["result_image"] = harmonized
        state["harmonized"] = True
        print(f"  Harmonization complete. Result: {harmonized.shape[1]}x{harmonized.shape[0]}")
    except Exception as e:
        logger.warning("Harmonization failed: %s. Keeping original stitched result.", e)
        state["harmonized"] = True  # Don't retry
        print(f"  Harmonization failed, keeping original: {e}")

    return state


def decide_node(state: AgentState) -> str:
    """Route after review: accept, harmonize, or error."""
    if state.get("error"):
        return "error"

    review = state.get("review_result", {})
    if review.get("approved", True):
        print("\n" + "=" * 60)
        print("  FINAL: Result accepted")
        print("=" * 60)
        return "done"

    if not state.get("harmonized", False):
        return "harmonize"

    # Already harmonized, accept whatever we have
    print("\n" + "=" * 60)
    print("  FINAL: Result accepted (after harmonization)")
    print("=" * 60)
    return "done"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_agent() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("load_image", load_image_node)
    workflow.add_node("decompose", decompose_node)
    workflow.add_node("segment", segment_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("stitch", stitch_node)
    workflow.add_node("review", review_node)
    workflow.add_node("harmonize", harmonize_node)

    workflow.set_entry_point("load_image")

    workflow.add_conditional_edges("load_image", _has_error, {
        "error": END,
        "ok": "decompose",
    })
    workflow.add_conditional_edges("decompose", _has_error, {
        "error": END,
        "ok": "segment",
    })
    workflow.add_conditional_edges("segment", _has_error, {
        "error": END,
        "ok": "generate",
    })
    workflow.add_conditional_edges("generate", _has_error, {
        "error": END,
        "ok": "stitch",
    })
    workflow.add_conditional_edges("stitch", _has_error, {
        "error": END,
        "ok": "review",
    })

    # Review routes to: done (approved), harmonize (needs fixing), error
    workflow.add_conditional_edges("review", decide_node, {
        "done": END,
        "harmonize": "harmonize",
        "error": END,
    })

    # After harmonize, review again (or error out)
    workflow.add_conditional_edges("harmonize", _has_error, {
        "error": END,
        "ok": "review",
    })

    return workflow.compile()


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

class ImageStitchAgent:
    """High-level interface for the image stitching agent."""

    def __init__(self):
        self.graph = build_agent()

    def run(self, image_path: str, instruction: str) -> dict:
        """Run the full agent pipeline."""
        initial_state: AgentState = {
            "user_instruction": instruction,
            "original_image_path": image_path,
            "original_image": None,
            "plan": None,
            "masks": {},
            "generated_images": {},
            "current_step_idx": 0,
            "result_image": None,
            "review_result": None,
            "harmonized": False,
            "error": None,
        }

        final_state = self.graph.invoke(initial_state)

        if final_state.get("error"):
            logger.error("Pipeline error: %s", final_state["error"])

        if final_state.get("result_image") is not None:
            out_path = OUTPUT_DIR / f"final_{int(time.time())}.png"
            cv2.imwrite(str(out_path), final_state["result_image"])
            logger.info("Final result saved to %s", out_path)
            final_state["output_path"] = str(out_path)

        return final_state


def run_agent(image_path: str, instruction: str) -> dict:
    """One-liner to run the agent."""
    agent = ImageStitchAgent()
    return agent.run(image_path, instruction)
