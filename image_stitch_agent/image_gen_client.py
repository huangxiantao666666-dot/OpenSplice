"""
Image generation client using DashScope SDK (Z-Image-Turbo / Wanx).
"""

import logging
import time
import os
import concurrent.futures
from typing import Optional

import numpy as np
import cv2
import requests
import dashscope
from dashscope.aigc.image_generation import ImageGeneration
from dashscope.api_entities.dashscope_response import Message

from .config import QWEN_IMAGE_API_KEY, QWEN_IMAGE_MODEL, DASHSCOPE_IMAGE_EDIT_MODEL, OUTPUT_DIR

logger = logging.getLogger(__name__)

# Configure DashScope
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

_API_TIMEOUT = 120  # seconds for API calls


def _run_with_timeout(fn, timeout=_API_TIMEOUT):
    """Run fn in a thread with timeout. Raises TimeoutError if it hangs."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        return future.result(timeout=timeout)


def generate_image(prompt: str, size: str = "1024*1024", n: int = 1) -> np.ndarray:
    """
    Generate an image using DashScope ImageGeneration API.

    Args:
        prompt: Text description of the image to generate.
        size: Image size, e.g. "1024*1024", "2K".
        n: Number of images to generate (returns the first).

    Returns:
        RGB image as numpy array (H, W, 3), uint8.
    """
    logger.info("Generating image with prompt: %s", prompt)

    message = Message(role="user", content=[{"text": prompt}])

    # Try sync call first (for faster models like z-image-turbo)
    response = _run_with_timeout(
        lambda: ImageGeneration.call(
            model=QWEN_IMAGE_MODEL,
            api_key=QWEN_IMAGE_API_KEY,
            messages=[message],
            n=n,
            size=size,
        )
    )

    if response.status_code == 200:
        return _extract_image(response, prompt)

    # If sync fails, try async
    logger.info("Sync call returned %s, trying async...", response.status_code)
    response = _run_with_timeout(
        lambda: ImageGeneration.async_call(
            model=QWEN_IMAGE_MODEL,
            api_key=QWEN_IMAGE_API_KEY,
            messages=[message],
            n=n,
            size=size,
        )
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Image generation failed: code={response.code}, message={response.message}"
        )

    task_id = response.output.task_id
    logger.info("Async task submitted: %s, waiting...", task_id)

    status = _run_with_timeout(
        lambda: ImageGeneration.wait(task=response, api_key=QWEN_IMAGE_API_KEY),
        timeout=300,  # async generation can take minutes
    )
    if status.output.task_status != "SUCCEEDED":
        raise RuntimeError(
            f"Image generation task failed: status={status.output.task_status}, "
            f"code={status.code}, message={status.message}"
        )

    return _extract_image(status, prompt)


def _extract_image(response, prompt: str) -> np.ndarray:
    """Extract image from DashScope response and return as RGB numpy array.

    Handles two response formats:
    - OpenAI-compatible: choices[0].message.content[0].image (URL)
    - Native DashScope: output.results[0].url
    """
    output = response.output

    # Try OpenAI-compatible format (used by z-image-turbo)
    choices = output.get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content", [])
        for item in content:
            if isinstance(item, dict) and item.get("image"):
                image_url = item["image"]
                logger.info("Downloading generated image from: %s", image_url[:80])
                img_resp = requests.get(image_url, timeout=60)
                img_resp.raise_for_status()
                nparr = np.frombuffer(img_resp.content, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                _save_generated(img, prompt)
                return img

    # Fallback: native DashScope format (output.results[0].url)
    results = output.get("results", [])
    if results:
        image_url = results[0].get("url")
        if image_url:
            logger.info("Downloading generated image from: %s", image_url[:80])
            img_resp = requests.get(image_url, timeout=60)
            img_resp.raise_for_status()
            nparr = np.frombuffer(img_resp.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            _save_generated(img, prompt)
            return img

    raise RuntimeError(f"No image found in response output: {dict(output)}")


def harmonize_image(image: np.ndarray, edit_prompt: str) -> np.ndarray:
    """
    Harmonize a stitched image in-place using Qwen Image Edit.

    Sends the composite image to the editing model with instructions to fix
    lighting, shadows, edges, and blend the pasted object into the scene.

    Args:
        image: RGB image (H, W, 3) as numpy array.
        edit_prompt: Instruction describing what to fix (e.g. "fix lighting").

    Returns:
        Harmonized RGB image.
    """
    from dashscope import MultiModalConversation
    import base64

    logger.info("Harmonizing image with edit prompt: %s", edit_prompt)

    # cv2.imencode expects BGR — convert if needed
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", img_bgr)
    b64 = base64.b64encode(buf).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64}"

    messages = [{
        "role": "user",
        "content": [
            {"image": data_uri},
            {"text": edit_prompt},
        ],
    }]

    response = _run_with_timeout(
        lambda: MultiModalConversation.call(
            api_key=QWEN_IMAGE_API_KEY,
            model=DASHSCOPE_IMAGE_EDIT_MODEL,
            messages=messages,
            n=1,
            watermark=False,
        )
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Image harmonization failed: code={response.code}, message={response.message}"
        )

    output = response.output
    choices = output.get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content", [])
        for item in content:
            if isinstance(item, dict) and item.get("image"):
                image_url = item["image"]
                logger.info("Downloading harmonized image from: %s", image_url[:80])
                img_resp = requests.get(image_url, timeout=60)
                img_resp.raise_for_status()
                nparr = np.frombuffer(img_resp.content, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                out_path = OUTPUT_DIR / "last_harmonized.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                logger.info("Saved harmonized image to %s", out_path)
                return img

    raise RuntimeError(f"No image in harmonization response: {dict(output)}")


def load_source_image(path: str) -> np.ndarray:
    """Load an image from a user-provided path. Returns RGB numpy array."""
    print(f"[DEBUG] load_source_image: path={path}, type={type(path).__name__}")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image from: {path}")
    print(f"[DEBUG] load_source_image: loaded {img.shape}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pose_adapt_image(
    context_image: np.ndarray,
    replacement_image: np.ndarray,
    edit_prompt: str,
) -> np.ndarray:
    """
    Send TWO images to Qwen-Image-Edit: a context image (original masked region
    showing the desired POSE/SHAPE) and a replacement image (showing the desired
    APPEARANCE). The model generates a new image combining both.

    Args:
        context_image: RGB image showing the target pose/position (mask region on black bg).
        replacement_image: RGB image of the replacement object (any pose).
        edit_prompt: Instruction, e.g. "make this dog lie down like the dog in image 1".

    Returns:
        Adapted RGB image.
    """
    from dashscope import MultiModalConversation
    import base64

    logger.info("Pose-adapting with prompt: %s", edit_prompt)

    # cv2.imencode expects BGR — convert both images
    ctx_bgr = cv2.cvtColor(context_image, cv2.COLOR_RGB2BGR)
    repl_bgr = cv2.cvtColor(replacement_image, cv2.COLOR_RGB2BGR)
    _, buf1 = cv2.imencode(".png", ctx_bgr)
    b64_ctx = base64.b64encode(buf1).decode("utf-8")
    _, buf2 = cv2.imencode(".png", repl_bgr)
    b64_repl = base64.b64encode(buf2).decode("utf-8")

    messages = [{
        "role": "user",
        "content": [
            {"image": f"data:image/png;base64,{b64_ctx}"},
            {"image": f"data:image/png;base64,{b64_repl}"},
            {"text": edit_prompt},
        ],
    }]

    response = _run_with_timeout(
        lambda: MultiModalConversation.call(
            api_key=QWEN_IMAGE_API_KEY,
            model=DASHSCOPE_IMAGE_EDIT_MODEL,
            messages=messages,
            n=1,
            watermark=False,
        )
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Pose adaptation failed: code={response.code}, message={response.message}"
        )

    output = response.output
    choices = output.get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content", [])
        for item in content:
            if isinstance(item, dict) and item.get("image"):
                image_url = item["image"]
                logger.info("Downloading pose-adapted image from: %s", image_url[:80])
                img_resp = requests.get(image_url, timeout=60)
                img_resp.raise_for_status()
                nparr = np.frombuffer(img_resp.content, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                out_path = OUTPUT_DIR / "last_pose_adapted.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                logger.info("Saved pose-adapted image to %s", out_path)
                return img

    raise RuntimeError(f"No image in pose adaptation response: {dict(output)}")


def _save_generated(img: np.ndarray, prompt: str):
    """Save generated image for debugging — overwrites last_generated.png."""
    out_path = OUTPUT_DIR / "last_generated.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    logger.info("Saved generated image to %s", out_path)
