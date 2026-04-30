from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timeline_align.keyframe_filter import (  # noqa: E402
    compute_change_score,
    compute_text_like_score,
    prepare_detection_gray,
)


def load_detection_gray(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Unable to read image: {image_path}")
    return prepare_detection_gray(image)


def compare_images(
    first_image: Path,
    second_image: Path,
    global_threshold: float = 12.0,
    subtitle_threshold: float = 8.0,
) -> dict[str, float | bool]:
    first_gray = load_detection_gray(first_image)
    second_gray = load_detection_gray(second_image)

    global_score = compute_change_score(second_gray, first_gray)
    text_like_score = compute_text_like_score(second_gray, first_gray)
    is_similar = (
        global_score < global_threshold and text_like_score < subtitle_threshold
    )
    return {
        "global_score": round(global_score, 3),
        "text_like_score": round(text_like_score, 3),
        "global_threshold": global_threshold,
        "subtitle_threshold": subtitle_threshold,
        "is_similar": is_similar,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two images using keyframe_filter.py similarity scores."
    )
    parser.add_argument("--first", default='', help="First image path.")
    parser.add_argument("--second", default='', help="Second image path.")
    parser.add_argument("--global-threshold", type=float, default=12.0)
    parser.add_argument("--subtitle-threshold", type=float, default=8.0)
    args = parser.parse_args()

    result = compare_images(
        first_image=Path(args.first),
        second_image=Path(args.second),
        global_threshold=args.global_threshold,
        subtitle_threshold=args.subtitle_threshold,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
