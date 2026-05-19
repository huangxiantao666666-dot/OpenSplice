"""
LLM API client: Qwen Vision for both task decomposition and visual feedback.
"""

import json
import logging
import base64
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image
from openai import OpenAI

from .config import (
    QWEN_API_KEY,
    QWEN_BASE_URL,
    QWEN_VISION_MODEL,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
    return _client


def _encode_b64(img: np.ndarray) -> str:
    """Convert BGR numpy image to base64 JPEG string."""
    rgb = img[:, :, ::-1] if img.shape[2] == 3 else img
    pil_img = Image.fromarray(rgb)
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _strip_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


# ---------------------------------------------------------------------------
# Task decomposition — vision model sees the original image
# ---------------------------------------------------------------------------

def decompose_task(user_instruction: str, image: np.ndarray) -> dict:
    """
    Send user instruction + original image to Qwen Vision,
    get a decomposed task plan in JSON.
    """
    client = _get_client()
    img_b64 = _encode_b64(image)

    logger.info("Decomposing instruction (with vision): %s", user_instruction)

    response = client.chat.completions.create(
        model=QWEN_VISION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": f"User instruction: {user_instruction}"},
                ],
            },
        ],
        temperature=0.1,
        max_tokens=4096,
    )

    raw = _strip_json(response.choices[0].message.content)
    logger.info("Task plan:\n%s", raw)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Visual quality check — compares original vs modified
# ---------------------------------------------------------------------------

def vision_check(
    original_image: np.ndarray,
    modified_image: np.ndarray,
    user_instruction: str,
    generation_prompt: str,
) -> dict:
    """
    Ask Qwen Vision to review the modified image and decide if it needs regeneration.

    Returns: {"approved": bool, "score": 1-10, "issues": [...], "feedback": str, "new_generation_prompt": str|null}
    """
    client = _get_client()

    original_b64 = _encode_b64(original_image)
    modified_b64 = _encode_b64(modified_image)

    review_prompt = f"""You are a quality inspector for an image editing pipeline.

Original user request: "{user_instruction}"
The generation prompt used was: "{generation_prompt}"

Review the modified image carefully:
1. Does the edit match what the user asked for?
2. Is the blending/seam between the generated content and the original image natural?
3. Is the generated content of good quality (realistic, correct perspective, proper lighting)?

Respond in JSON format:
{{
  "approved": true/false,
  "score": 1-10,
  "issues": ["list of specific problems if any"],
  "feedback": "detailed feedback",
  "new_generation_prompt": "improved prompt for regeneration, or null if approved"
}}

If approved is false, provide a specific, improved generation prompt that addresses the issues.
Output ONLY valid JSON."""

    response = client.chat.completions.create(
        model=QWEN_VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Original image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{original_b64}"}},
                    {"type": "text", "text": "Modified image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{modified_b64}"}},
                    {"type": "text", "text": review_prompt},
                ],
            }
        ],
        temperature=0.1,
        max_tokens=2048,
    )

    raw = _strip_json(response.choices[0].message.content)
    logger.info("Vision check result:\n%s", raw)
    return json.loads(raw)
