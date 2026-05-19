#!/usr/bin/env python
"""
Image Stitch Agent — Main Entry Point

Usage:
    python -m image_stitch_agent.main --image <path> --instruction "<text>"

Example:
    python -m image_stitch_agent.main \\
        --image "test.jpg" \\
        --instruction "把图像中的穿红色衣服的人换成一个猪头"
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure parent is on path for package imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from image_stitch_agent.workflow import ImageStitchAgent


def main():
    parser = argparse.ArgumentParser(
        description="Image Stitch Agent — modify images with natural language instructions"
    )
    parser.add_argument(
        "--image", "-i", required=True, help="Path to the input image"
    )
    parser.add_argument(
        "--instruction", "-p", required=True, help="Natural language editing instruction"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    agent = ImageStitchAgent()
    result = agent.run(
        image_path=args.image,
        instruction=args.instruction,
    )

    if result.get("error"):
        print(f"\nERROR: {result['error']}")
        sys.exit(1)

    print(f"\nDone! Output saved to: {result.get('output_path', 'unknown')}")

    review = result.get("review_result", {})
    if review:
        print(f"Quality score: {review.get('score', 'N/A')}/10")
        if review.get("feedback"):
            print(f"Feedback: {review['feedback']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
