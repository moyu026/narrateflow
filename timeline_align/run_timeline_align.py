from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timeline_align.keyframe_filter import sample_keyframes
from timeline_align.vl_client import (
    extract_frames_at_times,
    load_gemini_api_key,
    load_page_segments,
    load_selected_keyframes,
    probe_frames,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_video_context_timeline(spoken_json: Path) -> dict[str, Any]:
    payload = load_json(spoken_json)
    paragraphs = payload.get("paragraphs", [])
    segments: list[dict[str, Any]] = []
    voice_count = 0
    matched_count = 0
    for paragraph in paragraphs:
        is_silent = bool(paragraph.get("is_silent", False))
        is_cover = bool(paragraph.get("is_cover", False))
        matched = paragraph.get("start_time") is not None and paragraph.get("end_time") is not None
        if matched:
            matched_count += 1
        voice_count += 1
        segments.append(
            {
                "paragraph_index": int(paragraph["index"]),
                "spoken_text": paragraph.get("spoken_text", ""),
                "timeline_enabled": True,
                "matched": matched,
                "start": round(float(paragraph.get("start_time", 0.0)), 3) if matched else None,
                "end_hint": round(float(paragraph.get("end_time", 0.0)), 3) if matched else None,
                "review_status": "auto" if matched else "needs_manual",
                "is_silent": is_silent,
                "is_cover": is_cover,
                "source": "video_window" if not is_cover else "cover_window",
            }
        )
    return {
        "spoken_json": str(spoken_json),
        "page": int(payload.get("page", 1)),
        "status": "complete" if voice_count == matched_count else "incomplete",
        "voice_count": voice_count,
        "matched_count": matched_count,
        "missing_paragraph_indices": [
            int(item["paragraph_index"]) for item in segments if not item.get("matched")
        ],
        "cover_paragraph_index": payload.get("cover_paragraph_index"),
        "segments": segments,
        "notes": [
            "video_context_windows 模式直接信任 spoken.json 中的 start_time/end_time。",
            "is_cover 段用于封面语音前缀；主视频时间轴从后续窗口开始。",
            "is_silent 段保留时间窗，但不会在 TTS 阶段生成音频。",
        ],
    }


def auto_select_probe_times(
    keyframes_json: Path, desired_count: int = 8
) -> list[float]:
    payload = load_json(keyframes_json)
    candidates = payload.get("candidates", [])
    if not candidates:
        return []

    priority = {
        "text_like_change": 0,
        "scene_change": 1,
        "stable_fill": 2,
    }
    desired_count = max(3, min(desired_count, len(candidates)))
    buckets: list[list[dict]] = [[] for _ in range(desired_count)]
    for idx, candidate in enumerate(candidates):
        bucket_idx = min(desired_count - 1, int(idx * desired_count / len(candidates)))
        buckets[bucket_idx].append(candidate)

    selected: list[float] = []
    for bucket in buckets:
        if not bucket:
            continue
        best = sorted(
            bucket,
            key=lambda item: (
                priority.get(item.get("type", "stable_fill"), 99),
                -float(item.get("text_like_score", 0.0)),
                -float(item.get("global_score", 0.0)),
            ),
        )[0]
        selected.append(round(float(best["time"]), 2))
    return sorted({value for value in selected})


def build_probe_payload(
    mode: str,
    spoken_json: Path,
    output_path: Path,
    api_key: str | None = None,
    video: str | None = None,
    keyframes_json: str | None = None,
    times_text: str | None = None,
    paragraphs_text: str | None = None,
    frame_hint: str | None = None,
    body_start_paragraph_index: int = 2,
) -> dict[str, Any]:
    resolved_api_key = load_gemini_api_key(api_key)
    title, segments = load_page_segments(spoken_json)
    segments = [
        s for s in segments if int(s.get("paragraph_index", 0)) >= body_start_paragraph_index
    ]

    if paragraphs_text:
        paragraph_indices = {
            int(x.strip()) for x in paragraphs_text.split(",") if x.strip()
        }
        segments = [
            s for s in segments if int(s["paragraph_index"]) in paragraph_indices
        ]

    if mode == "keyframes":
        if not keyframes_json:
            raise SystemExit("--keyframes-json is required for mode=keyframes")
        if times_text:
            times = [float(x.strip()) for x in times_text.split(",") if x.strip()]
        else:
            times = auto_select_probe_times(Path(keyframes_json))
            if not times:
                raise SystemExit("No usable keyframes found for automatic probing")
        frames = load_selected_keyframes(Path(keyframes_json), times)
        hint_builder = (
            lambda frame: f"当前视频时间点约为 {frame['time']} 秒。请在候选段落中判断。"
        )
    else:
        if not video or not times_text:
            raise SystemExit("--video and --times are required for mode=times")
        times = [float(x.strip()) for x in times_text.split(",") if x.strip()]
        frame_dir = output_path.parent / "probe_frames"
        frames = extract_frames_at_times(Path(video), frame_dir, times)
        hint_builder = (
            lambda frame: frame_hint
            or f"当前视频时间点约为 {frame['time']} 秒。请在候选段落中判断。"
        )

    results = probe_frames(
        api_key=resolved_api_key,
        title=title,
        segments=segments,
        frames=frames,
        frame_hint_builder=hint_builder,
    )
    return {"mode": mode, "spoken_json": str(spoken_json), "results": results}


def summarize_results(results: list[dict], min_confidence: float) -> list[dict]:
    accepted = [
        item
        for item in results
        if item.get("confidence", 0) >= min_confidence
        and isinstance(item.get("best_paragraph_index"), int)
    ]
    windows: list[dict] = []
    current: dict | None = None
    for item in accepted:
        paragraph_index = item["best_paragraph_index"]
        if current is None or current["paragraph_index"] != paragraph_index:
            if current is not None:
                windows.append(current)
            current = {
                "paragraph_index": paragraph_index,
                "start": item["time"],
                "end": item["time"],
                "samples": [item],
            }
        else:
            current["end"] = item["time"]
            current["samples"].append(item)
    if current is not None:
        windows.append(current)
    return windows


def monotonic_filter(results: list[dict], min_confidence: float) -> list[dict]:
    filtered: list[dict] = []
    best_so_far = 0
    for item in results:
        idx = item.get("best_paragraph_index")
        conf = float(item.get("confidence", 0.0) or 0.0)
        item = dict(item)
        if not (conf >= min_confidence and isinstance(idx, int)):
            item["monotonic_kept"] = False
            filtered.append(item)
            continue
        if idx < best_so_far:
            item["monotonic_kept"] = False
            item["rejected_reason"] = "backtrack"
            filtered.append(item)
            continue
        item["monotonic_kept"] = True
        best_so_far = max(best_so_far, idx)
        filtered.append(item)
    return filtered


def refine_windows(windows: list[dict], step_sec: float) -> list[dict]:
    pad = step_sec / 2.0
    return [
        {
            **w,
            "start": max(0.0, round(w["start"] - pad, 2)),
            "end": round(w["end"] + pad, 2),
        }
        for w in windows
    ]


def build_rough(probe_payload: dict, min_confidence: float, step_sec: float) -> dict:
    results = probe_payload.get("results", [])
    raw_windows = summarize_results(results, min_confidence=min_confidence)
    monotonic_results = monotonic_filter(results, min_confidence=min_confidence)
    kept = [item for item in monotonic_results if item.get("monotonic_kept")]
    windows = summarize_results(kept, min_confidence=min_confidence)
    refined_windows = refine_windows(windows, step_sec=step_sec)
    return {
        "mode": probe_payload.get("mode"),
        "spoken_json": probe_payload.get("spoken_json"),
        "results": monotonic_results,
        "raw_windows": raw_windows,
        "windows": windows,
        "refined_windows": refined_windows,
    }


def build_final(
    spoken_json: Path,
    anchors_payloads: list[dict[str, Any]],
    body_start_paragraph_index: int = 2,
) -> dict[str, Any]:
    spoken_payload = load_json(spoken_json)
    paragraph_texts = {1: spoken_payload.get("title_text", "")}
    for item in spoken_payload.get("paragraphs", []):
        paragraph_texts[int(item["index"])] = item["spoken_text"]

    anchor_map: dict[int, dict] = {}
    for payload in anchors_payloads:
        windows = payload.get("refined_windows") or payload.get("windows") or []
        if not windows and payload.get("results"):
            grouped: dict[int, dict] = {}
            for item in payload["results"]:
                idx = item.get("best_paragraph_index")
                if not isinstance(idx, int):
                    continue
                t = float(item["time"])
                current = grouped.get(idx)
                if current is None:
                    grouped[idx] = {"paragraph_index": idx, "start": t, "end": t}
                else:
                    current["start"] = min(current["start"], t)
                    current["end"] = max(current["end"], t)
            windows = list(grouped.values())

        for window in windows:
            idx = int(window["paragraph_index"])
            start = float(window["start"])
            end = float(window["end"])
            current = anchor_map.get(idx)
            if current is None:
                anchor_map[idx] = {
                    "paragraph_index": idx,
                    "anchor_start": start,
                    "anchor_end": end,
                    "candidate_starts": [start],
                    "candidate_ends": [end],
                    "source_files": [payload.get("source_name", "runtime")],
                }
            else:
                current["anchor_start"] = min(current["anchor_start"], start)
                current["anchor_end"] = max(current["anchor_end"], end)
                current.setdefault("candidate_starts", []).append(start)
                current.setdefault("candidate_ends", []).append(end)
                current.setdefault("source_files", []).append(
                    payload.get("source_name", "runtime")
                )

    for item in anchor_map.values():
        item["candidate_starts"] = sorted(
            {round(float(value), 3) for value in item.get("candidate_starts", [])}
        )
        item["candidate_ends"] = sorted(
            {round(float(value), 3) for value in item.get("candidate_ends", [])}
        )

    timeline = []
    for idx in range(1, max(paragraph_texts.keys()) + 1):
        timeline_enabled = idx == 1 or idx >= body_start_paragraph_index
        item = {
            "paragraph_index": idx,
            "role": "title" if idx == 1 else "voice",
            "spoken_text": paragraph_texts.get(idx, ""),
            "timeline_enabled": timeline_enabled,
            "matched": timeline_enabled and idx in anchor_map,
        }
        if idx not in {1} and not timeline_enabled:
            item["review_status"] = "cover_intro"
        if timeline_enabled and idx in anchor_map:
            item.update(anchor_map[idx])
        timeline.append(item)

    payload = {
        "spoken_json": str(spoken_json),
        "page": spoken_payload.get("page", 1),
        "timeline": timeline,
        "notes": [
            "paragraph 1 是标题，不参与配音。",
            f"paragraph_index < {body_start_paragraph_index} 的正文段当前不参与正文时间轴对齐。",
            "其余段落的 anchor_start/anchor_end 为当前阶段汇总后的正式锚点范围。",
            "该时间轴允许静默区，不要求整段视频都填满音频。",
        ],
    }
    payload.update(build_timeline_status(payload))
    return payload


def build_public_timeline(
    final_payload: dict[str, Any], spoken_json: Path
) -> dict[str, Any]:
    public_segments = []
    for item in final_payload.get("timeline", []):
        if item.get("role") != "voice":
            continue
        public_segments.append(
            {
                "paragraph_index": int(item["paragraph_index"]),
                "spoken_text": item.get("spoken_text", ""),
                "timeline_enabled": bool(item.get("timeline_enabled", True)),
                "matched": bool(item.get("matched", False)),
                "start": round(float(item["anchor_start"]), 3)
                if item.get("matched")
                else None,
                "end_hint": round(float(item["anchor_end"]), 3)
                if item.get("matched")
                else None,
                "review_status": item.get("review_status")
                or ("needs_manual" if not item.get("matched") else "auto"),
            }
        )
    return {
        "spoken_json": str(spoken_json),
        "page": final_payload.get("page", 1),
        "status": final_payload.get("status"),
        "voice_count": final_payload.get("voice_count"),
        "matched_count": final_payload.get("matched_count"),
        "missing_paragraph_indices": final_payload.get("missing_paragraph_indices", []),
        "segments": public_segments,
        "notes": [
            "start 是当前段的主接入时间。",
            "end_hint 仅为参考结束窗口，不作为音频裁剪硬边界。",
            "实际播放区间由 Stage4 根据音频时长和 buffer 决定。",
        ],
    }


def enforce_monotonic_starts(
    final_payload: dict[str, Any], min_step_sec: float = 0.2
) -> dict[str, Any]:
    timeline = [dict(item) for item in final_payload.get("timeline", [])]
    previous_start: float | None = None
    conflict_gap_sec = 1.0
    for item in timeline:
        if item.get("role") != "voice" or not item.get("matched"):
            continue
        current_start = float(item["anchor_start"])
        if previous_start is not None and current_start < previous_start + min_step_sec:
            item["original_anchor_start"] = current_start
            candidates = [
                float(value)
                for value in item.get("candidate_starts", [])
                if float(value) >= previous_start + min_step_sec
            ]
            if candidates:
                item["anchor_start"] = round(min(candidates), 3)
            else:
                item["anchor_start"] = round(previous_start + min_step_sec, 3)
            if "anchor_end" in item and item["anchor_end"] is not None:
                current_end = float(item["anchor_end"])
                item["anchor_end"] = round(max(current_end, item["anchor_start"]), 3)
            if item.get("review_status") in (None, "auto"):
                item["review_status"] = "auto_order_adjusted"
        elif (
            previous_start is not None
            and current_start < previous_start + conflict_gap_sec
        ):
            candidates = [
                float(value)
                for value in item.get("candidate_starts", [])
                if float(value) >= previous_start + conflict_gap_sec
            ]
            if candidates:
                item["original_anchor_start"] = current_start
                item["anchor_start"] = round(min(candidates), 3)
                if item.get("review_status") in (None, "auto"):
                    item["review_status"] = "auto_order_adjusted"
        previous_start = float(item["anchor_start"])

    payload = {**final_payload, "timeline": timeline}
    payload.update(build_timeline_status(payload))
    return payload


def build_timeline_status(final_payload: dict[str, Any]) -> dict[str, Any]:
    timeline = final_payload.get("timeline", [])
    voice_items = [
        item
        for item in timeline
        if item.get("role") == "voice" and item.get("timeline_enabled", True)
    ]
    matched_items = [item for item in voice_items if item.get("matched")]
    missing_indices = [
        int(item["paragraph_index"]) for item in voice_items if not item.get("matched")
    ]
    voice_count = len(voice_items)
    matched_count = len(matched_items)
    if voice_count == matched_count:
        status = "complete"
    elif voice_count - matched_count <= 2:
        status = "near_complete"
    else:
        status = "incomplete"
    return {
        "status": status,
        "voice_count": voice_count,
        "matched_count": matched_count,
        "missing_paragraph_indices": missing_indices,
    }


def collect_missing_ranges(
    final_payload: dict[str, Any], video_duration: float, start_only: bool = False
) -> list[dict[str, Any]]:
    timeline = final_payload.get("timeline", [])
    missing_ranges: list[dict[str, Any]] = []
    voice_items = [
        item
        for item in timeline
        if item.get("role") == "voice" and item.get("timeline_enabled", True)
    ]
    for index, item in enumerate(voice_items):
        if item.get("matched"):
            continue
        paragraph_index = int(item["paragraph_index"])

        prev_item = None
        for j in range(index - 1, -1, -1):
            candidate = voice_items[j]
            if candidate.get("matched"):
                prev_item = candidate
                break

        next_item = None
        for j in range(index + 1, len(voice_items)):
            candidate = voice_items[j]
            if candidate.get("matched"):
                next_item = candidate
                break

        if prev_item:
            start = (
                float(prev_item["anchor_start"])
                if start_only
                else float(prev_item["anchor_end"])
            )
        else:
            start = 0.0
        end = float(next_item["anchor_start"]) if next_item else video_duration
        if end <= start + 0.8:
            continue
        missing_ranges.append(
            {
                "paragraph_index": paragraph_index,
                "start": round(start, 2),
                "end": round(end, 2),
            }
        )
    return missing_ranges


def build_gap_probe_times(start: float, end: float, limit: int = 3) -> list[float]:
    if end <= start:
        return []
    span = end - start
    count = min(limit, max(1, int(span // 6) + 1))
    step = span / (count + 1)
    return [round(start + step * i, 2) for i in range(1, count + 1)]


def select_gap_probe_times(
    keyframes_json: Path,
    start: float,
    end: float,
    limit: int = 3,
) -> list[float]:
    if end <= start:
        return []
    payload = load_json(keyframes_json)
    candidates = payload.get("candidates", [])
    priority = {
        "text_like_change": 0,
        "scene_change": 1,
        "stable_fill": 2,
    }
    gap_candidates = [
        item
        for item in candidates
        if start <= float(item.get("time", -1)) <= end
    ]
    midpoint = (start + end) / 2.0
    ranked = sorted(
        gap_candidates,
        key=lambda item: (
            priority.get(item.get("type", "stable_fill"), 99),
            abs(float(item.get("time", 0.0)) - midpoint),
            -float(item.get("text_like_score", 0.0)),
            -float(item.get("global_score", 0.0)),
        ),
    )
    selected = [round(float(item["time"]), 2) for item in ranked[:limit]]
    if len(selected) >= limit:
        return sorted(dict.fromkeys(selected))
    fallback = build_gap_probe_times(start, end, limit=limit)
    return sorted(dict.fromkeys(selected + fallback))[:limit]


def run_gap_reprobe(
    video: Path,
    spoken_json: Path,
    final_payload: dict[str, Any],
    debug_dir: Path,
    api_key: str | None,
    video_duration: float,
    round_index: int,
    keyframes_json: Path | None = None,
    start_only: bool = False,
) -> list[dict[str, Any]]:
    gap_payloads: list[dict[str, Any]] = []
    voice_timeline = [
        item
        for item in final_payload.get("timeline", [])
        if item.get("role") == "voice" and item.get("timeline_enabled", True)
    ]
    missing_ranges = collect_missing_ranges(
        final_payload, video_duration=video_duration, start_only=start_only
    )
    if len(missing_ranges) > 3:
        missing_ranges = missing_ranges[:3]
    for item in missing_ranges:
        if keyframes_json is not None and keyframes_json.exists():
            times = select_gap_probe_times(
                keyframes_json=keyframes_json,
                start=item["start"],
                end=item["end"],
            )
        else:
            times = build_gap_probe_times(item["start"], item["end"])
        if not times:
            continue
        current_idx = int(item["paragraph_index"])
        candidate_indices = {current_idx}
        for voice_item in voice_timeline:
            idx = int(voice_item["paragraph_index"])
            if abs(idx - current_idx) <= 1:
                candidate_indices.add(idx)
        payload = build_probe_payload(
            mode="times",
            spoken_json=spoken_json,
            output_path=debug_dir
            / f"gap_r{round_index:02d}_p{current_idx:02d}.probe.json",
            api_key=api_key,
            video=str(video),
            times_text=",".join(str(t) for t in times),
            paragraphs_text=",".join(str(idx) for idx in sorted(candidate_indices)),
            frame_hint=(
                f"请重点判断 paragraph_index={current_idx} 是否出现在这段时间范围内。若无法判断可返回 unknown。"
            ),
        )
        payload["source_name"] = f"gap_r{round_index:02d}_p{current_idx:02d}.probe.json"
        gap_payloads.append(payload)
        write_json(debug_dir / payload["source_name"], payload)
    return gap_payloads


def run_timeline_align(
    video: Path,
    spoken_json: Path,
    output: Path,
    debug_dir: Path | None = None,
    api_key: str | None = None,
    probe_mode: str = "keyframes",
    probe_times: str | None = None,
    probe_paragraphs: str | None = None,
    min_confidence: float = 0.65,
    step_sec: float = 4.0,
    fps_sample: float = 1.0,
    min_gap_sec: float = 1.5,
    global_threshold: float = 8.0,
    subtitle_threshold: float = 5.5,
    skip_keyframes: bool = False,
    gap_start_only: bool = True,
    cover_paragraph_index: int | None = None,
) -> dict[str, Any]:
    spoken_payload = load_json(spoken_json)
    if spoken_payload.get("source_type") == "video_context_windows":
        public_payload = build_video_context_timeline(spoken_json)
        write_json(output, public_payload)
        return public_payload

    body_start_paragraph_index = (
        int(cover_paragraph_index) + 1 if cover_paragraph_index is not None else 2
    )
    page_stem = output.stem.replace(".timeline", "")
    resolved_debug_dir = debug_dir or output.parent / "debug" / page_stem
    resolved_debug_dir.mkdir(parents=True, exist_ok=True)

    keyframes_json = resolved_debug_dir / "keyframes.json"
    probe_json = resolved_debug_dir / f"{page_stem}.probe.json"
    rough_json = resolved_debug_dir / f"{page_stem}.rough.json"

    if not skip_keyframes:
        keyframes_payload = sample_keyframes(
            video_path=video,
            out_dir=resolved_debug_dir / "keyframes",
            fps_sample=fps_sample,
            min_gap_sec=min_gap_sec,
            global_threshold=global_threshold,
            subtitle_threshold=subtitle_threshold,
        )
        write_json(keyframes_json, keyframes_payload)
    elif not keyframes_json.exists() and probe_mode == "keyframes":
        raise FileNotFoundError(f"keyframes json not found: {keyframes_json}")

    probe_payload = build_probe_payload(
        mode=probe_mode,
        spoken_json=spoken_json,
        output_path=probe_json,
        api_key=api_key,
        video=str(video),
        keyframes_json=str(keyframes_json) if probe_mode == "keyframes" else None,
        times_text=probe_times,
        paragraphs_text=probe_paragraphs,
        body_start_paragraph_index=body_start_paragraph_index,
    )
    write_json(probe_json, probe_payload)

    rough_payload = build_rough(
        probe_payload, min_confidence=min_confidence, step_sec=step_sec
    )
    rough_payload["source_name"] = rough_json.name
    write_json(rough_json, rough_payload)

    video_duration = (
        float(load_json(keyframes_json)["duration"]) if keyframes_json.exists() else 0.0
    )
    anchor_payloads: list[dict[str, Any]] = [rough_payload]
    final_payload = build_final(
        spoken_json=spoken_json,
        anchors_payloads=anchor_payloads,
        body_start_paragraph_index=body_start_paragraph_index,
    )
    reprobe_rounds: list[dict[str, Any]] = []

    for round_index in range(1, 4):
        before_missing = list(final_payload.get("missing_paragraph_indices", []))
        if not before_missing:
            break
        try:
            gap_payloads = run_gap_reprobe(
                video=video,
                spoken_json=spoken_json,
                final_payload=final_payload,
                debug_dir=resolved_debug_dir,
                api_key=api_key,
                video_duration=video_duration,
                round_index=round_index,
                keyframes_json=keyframes_json,
                start_only=gap_start_only,
            )
        except Exception as exc:
            reprobe_rounds.append(
                {
                    "round": round_index,
                    "before_missing": before_missing,
                    "after_missing": before_missing,
                    "newly_matched": [],
                    "probe_files": [],
                    "error": str(exc),
                    "status": "failed_but_kept_best_effort",
                }
            )
            break
        if not gap_payloads:
            reprobe_rounds.append(
                {
                    "round": round_index,
                    "before_missing": before_missing,
                    "after_missing": before_missing,
                    "newly_matched": [],
                    "probe_files": [],
                    "status": "no_gap_payloads",
                }
            )
            break

        gap_rough_payloads = []
        for gap_payload in gap_payloads:
            gap_rough = build_rough(
                gap_payload, min_confidence=min_confidence, step_sec=step_sec
            )
            gap_rough["source_name"] = str(
                gap_payload.get("source_name", "gap.probe.json")
            ).replace(".probe.json", ".rough.json")
            gap_rough_payloads.append(gap_rough)
            write_json(resolved_debug_dir / gap_rough["source_name"], gap_rough)

        anchor_payloads.extend(gap_rough_payloads)
        next_final_payload = build_final(
            spoken_json=spoken_json,
            anchors_payloads=anchor_payloads,
            body_start_paragraph_index=body_start_paragraph_index,
        )
        after_missing = list(next_final_payload.get("missing_paragraph_indices", []))
        newly_matched = [idx for idx in before_missing if idx not in after_missing]
        reprobe_rounds.append(
            {
                "round": round_index,
                "before_missing": before_missing,
                "after_missing": after_missing,
                "newly_matched": newly_matched,
                "probe_files": [payload.get("source_name") for payload in gap_payloads],
                "status": "completed",
            }
        )
        final_payload = next_final_payload
        if not newly_matched:
            break

    final_payload["reprobe_rounds"] = reprobe_rounds
    final_payload = enforce_monotonic_starts(final_payload)

    debug_output = output.with_suffix(output.suffix + ".debug.json")
    public_payload = build_public_timeline(final_payload, spoken_json=spoken_json)
    write_json(debug_output, final_payload)
    write_json(output, public_payload)
    return public_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 timeline alignment")
    parser.add_argument("--video", required=True)
    parser.add_argument("--spoken-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--debug-dir", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--probe-mode", choices=["keyframes", "times"], default="keyframes"
    )
    parser.add_argument("--probe-times", required=True)
    parser.add_argument("--probe-paragraphs", default=None)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--step-sec", type=float, default=4.0)
    parser.add_argument("--fps-sample", type=float, default=1.0)
    parser.add_argument("--min-gap-sec", type=float, default=1.5)
    parser.add_argument("--global-threshold", type=float, default=8.0)
    parser.add_argument("--subtitle-threshold", type=float, default=5.5)
    parser.add_argument("--skip-keyframes", action="store_true")
    parser.add_argument("--gap-start-only", action="store_true")
    parser.add_argument("--disable-gap-start-only", action="store_true")
    parser.add_argument("--cover-paragraph-index", type=int, default=None)
    args = parser.parse_args()

    result = run_timeline_align(
        video=Path(args.video),
        spoken_json=Path(args.spoken_json),
        output=Path(args.output),
        debug_dir=Path(args.debug_dir) if args.debug_dir else None,
        api_key=args.api_key,
        probe_mode=args.probe_mode,
        probe_times=args.probe_times,
        probe_paragraphs=args.probe_paragraphs,
        min_confidence=args.min_confidence,
        step_sec=args.step_sec,
        fps_sample=args.fps_sample,
        min_gap_sec=args.min_gap_sec,
        global_threshold=args.global_threshold,
        subtitle_threshold=args.subtitle_threshold,
        skip_keyframes=args.skip_keyframes,
        gap_start_only=(False if args.disable_gap_start_only else True),
        cover_paragraph_index=args.cover_paragraph_index,
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "matched": result.get("matched_count"),
                "status": result.get("status"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
