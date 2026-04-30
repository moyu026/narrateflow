from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract_all_frames(video_path: Path, output_dir: Path, image_ext: str = "png") -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            image_path = output_dir / f"frame_{frame_count:06d}.{image_ext}"
            if not cv2.imwrite(str(image_path), frame):
                raise RuntimeError(f"Unable to write frame image: {image_path}")
            frame_count += 1
    finally:
        cap.release()

    return frame_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract every frame from a video.")
    parser.add_argument("--video", default='', help="Input video path.")
    parser.add_argument("--output-dir", default='', help="Directory for frame images.")
    parser.add_argument(
        "--image-ext",
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Output image extension.",
    )
    args = parser.parse_args()

    count = extract_all_frames(
        video_path=Path(args.video),
        output_dir=Path(args.output_dir),
        image_ext=args.image_ext,
    )
    print(f"extracted_frames: {count}")
    print(f"output_dir: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
