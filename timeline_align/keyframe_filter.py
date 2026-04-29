from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm


def compute_change_score(curr: np.ndarray, prev: np.ndarray) -> float:
    diff = cv2.absdiff(curr, prev)
    return float(np.mean(diff))


def extract_text_like_mask(gray: np.ndarray) -> np.ndarray:
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.convertScaleAbs(
        cv2.addWeighted(cv2.absdiff(grad_x, 0), 1.0, cv2.absdiff(grad_y, 0), 1.0, 0)
    )
    _, thresh = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return closed


def compute_text_like_score(curr: np.ndarray, prev: np.ndarray) -> float:
    curr_mask = extract_text_like_mask(curr)
    prev_mask = extract_text_like_mask(prev)
    diff = cv2.absdiff(curr_mask, prev_mask)
    return float(np.mean(diff))


def insert_stable_fill_candidates(
    candidates: list[dict],
    cap: cv2.VideoCapture,
    fps: float,
    out_dir: Path,
    fill_gap_sec: float = 6.0,
) -> list[dict]:
    if len(candidates) < 2:
        return candidates

    enriched: list[dict] = []
    for index, current in enumerate(candidates):
        enriched.append(current)
        if index == len(candidates) - 1:
            continue
        nxt = candidates[index + 1]
        gap = float(nxt["time"] - current["time"])
        if gap < fill_gap_sec:
            continue
        fill_count = 1 if gap < fill_gap_sec * 1.8 else 2
        for fill_index in range(fill_count):
            ratio = (fill_index + 1) / (fill_count + 1)
            t = round(float(current["time"]) + gap * ratio, 2)
            frame_idx = int(round(t * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            image_path = out_dir / f"kf_{t:07.2f}.png"
            cv2.imwrite(str(image_path), frame)
            enriched.append(
                {
                    "time": t,
                    "frame_index": frame_idx,
                    "image_path": str(image_path),
                    "global_score": 0.0,
                    "text_like_score": 0.0,
                    "reason": ["stable_fill"],
                    "type": "stable_fill",
                }
            )
    return sorted(enriched, key=lambda item: float(item["time"]))


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
    sampled_steps = max(1, int(np.ceil(frame_count / stride))) if frame_count else None

    out_dir.mkdir(parents=True, exist_ok=True)

    prev_gray = None
    last_keep_frame = -(10**9)
    candidates: list[dict] = []
    index = 0
    progress = tqdm(
        total=sampled_steps,
        desc="Extracting keyframes",
        unit="sample",
        leave=False,
    )

    try:
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
                text_like_score = 999.0
            else:
                global_score = compute_change_score(gray, prev_gray)
                text_like_score = compute_text_like_score(gray, prev_gray)

            should_keep = False
            reason = []
            if index - last_keep_frame >= min_gap_frames:
                if global_score >= global_threshold:
                    should_keep = True
                    reason.append("global_change")
                if text_like_score >= subtitle_threshold:
                    should_keep = True
                    reason.append("text_like_change")

            if should_keep:
                timestamp = round(index / fps, 2)
                image_path = out_dir / f"kf_{timestamp:07.2f}.png"
                cv2.imwrite(str(image_path), frame)
                frame_type = (
                    "text_like_change" if "text_like_change" in reason else "scene_change"
                )
                candidates.append(
                    {
                        "time": timestamp,
                        "frame_index": index,
                        "image_path": str(image_path),
                        "global_score": round(global_score, 3),
                        "text_like_score": round(text_like_score, 3),
                        "reason": reason,
                        "type": frame_type,
                    }
                )
                last_keep_frame = index

            prev_gray = gray
            index += 1
            progress.update(1)
    finally:
        progress.close()

    candidates = insert_stable_fill_candidates(candidates, cap, fps, out_dir)
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
