from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def compute_change_score(curr: np.ndarray, prev: np.ndarray) -> float:
    diff = cv2.absdiff(curr, prev)
    return float(np.mean(diff))


def compute_subtitle_score(curr: np.ndarray, prev: np.ndarray) -> float:
    h, w = curr.shape[:2]
    y0 = int(h * 0.78)
    subtitle_curr = curr[y0:h, 0:w]
    subtitle_prev = prev[y0:h, 0:w]
    diff = cv2.absdiff(subtitle_curr, subtitle_prev)
    return float(np.mean(diff))


def sample_keyframes(
    video_path: Path,
    out_dir: Path,
    fps_sample: float = 1.0,
    min_gap_sec: float = 2.0,
    global_threshold: float = 12.0,
    subtitle_threshold: float = 8.0,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps else 0.0
    stride = max(1, int(round(fps / fps_sample)))
    min_gap_frames = int(round(min_gap_sec * fps))

    out_dir.mkdir(parents=True, exist_ok=True)

    prev_gray = None
    last_keep_frame = -(10**9)
    candidates: list[dict] = []
    index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride != 0:
            index += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_gray is None:
            global_score = 999.0
            subtitle_score = 999.0
        else:
            global_score = compute_change_score(gray, prev_gray)
            subtitle_score = compute_subtitle_score(gray, prev_gray)

        should_keep = False
        reason = []
        if index - last_keep_frame >= min_gap_frames:
            if global_score >= global_threshold:
                should_keep = True
                reason.append("global_change")
            if subtitle_score >= subtitle_threshold:
                should_keep = True
                reason.append("subtitle_change")

        if should_keep:
            timestamp = round(index / fps, 2)
            image_path = out_dir / f"kf_{timestamp:07.2f}.png"
            cv2.imwrite(str(image_path), frame)
            candidates.append(
                {
                    "time": timestamp,
                    "frame_index": index,
                    "image_path": str(image_path),
                    "global_score": round(global_score, 3),
                    "subtitle_score": round(subtitle_score, 3),
                    "reason": reason,
                }
            )
            last_keep_frame = index

        prev_gray = gray
        index += 1

    cap.release()
    return {
        "video_path": str(video_path),
        "duration": round(duration, 2),
        "fps": fps,
        "fps_sample": fps_sample,
        "min_gap_sec": min_gap_sec,
        "global_threshold": global_threshold,
        "subtitle_threshold": subtitle_threshold,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter keyframes for UI tutorial videos"
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps-sample", type=float, default=1.0)
    parser.add_argument("--min-gap-sec", type=float, default=2.0)
    parser.add_argument("--global-threshold", type=float, default=12.0)
    parser.add_argument("--subtitle-threshold", type=float, default=8.0)
    args = parser.parse_args()

    payload = sample_keyframes(
        video_path=Path(args.video),
        out_dir=Path(args.output).parent / "keyframes",
        fps_sample=args.fps_sample,
        min_gap_sec=args.min_gap_sec,
        global_threshold=args.global_threshold,
        subtitle_threshold=args.subtitle_threshold,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
