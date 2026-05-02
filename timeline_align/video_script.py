from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from timeline_align.keyframe_filter import sample_keyframes
from timeline_align.vl_client import (
    call_vl_gemini,
    load_gemini_api_key,
    parse_gemini_batch_response,
    safe_parse_json_from_content,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_reference_text(reference_text: Path | None) -> str:
    if reference_text is None or not reference_text.exists():
        return ""
    return reference_text.read_text(encoding="utf-8").strip()


def build_global_summary(
    cover_image: Path | None,
    reference_text: str,
) -> str:
    if reference_text:
        snippet = reference_text.replace("\r", " ").replace("\n", " ").strip()
        return f"参考资料摘要：{snippet}"
    if cover_image is not None:
        return f"视频包含封面图 {cover_image.stem}，请保持讲解直接、连贯。"
    return "视频内容将按关键画面生成讲解，请保持术语一致并直接进入操作说明。"


def normalize_video_title(video_stem: str) -> str:
    return "视频讲解"


def build_cover_prompt(reference_text: str) -> str:
    lines = [
        "你正在为一个视频撰写简短的封面页旁白。",
        "只返回 JSON，不要输出额外说明。",
        "spoken_text 要简洁、自然，适合直接朗读。",
    ]
    if reference_text:
        lines.append(f"参考文本：{reference_text}")
    lines.append(
        json.dumps(
            {
                "global_summary": "简要概括视频主题。",
                "cover_title": "视频讲解",
                "spoken_text": "这里是本视频的开场背景。",
                "reference_terms_used": [],
            },
            ensure_ascii=False,
        )
    )
    return "\n".join(lines)


def build_previous_context(previous_item: dict[str, Any] | None) -> tuple[str, str]:
    if previous_item is None:
        return "[START_OF_VIDEO]", "start"
    if previous_item.get("is_silent"):
        return f"[SILENT_GAP: {float(previous_item['silent_duration']):.1f}s]", "silent_gap"
    return str(previous_item.get("spoken_text", "")).strip(), "spoken_text"


def _window_has_meaningful_change(window: dict[str, Any]) -> bool:
    frames = window.get("frames", [])
    for frame in frames:
        if float(frame.get("global_score", 0.0)) >= 12.0:
            return True
        if float(frame.get("text_like_score", 0.0)) >= 8.0:
            return True
    return False


def build_cover_window(
    cover_image: Path,
    cover_duration_sec: float,
) -> dict[str, Any]:
    return {
        "window_id": "cover_0",
        "start_time": 0.0,
        "end_time": round(float(cover_duration_sec), 3),
        "is_cover": True,
        "frames": [{"time": 0.0, "image_path": str(cover_image)}],
    }


def build_video_windows(
    keyframes_payload: dict[str, Any],
    min_window_sec: float = 2.5,
    max_window_sec: float = 6.0,
) -> list[dict[str, Any]]:
    duration = float(keyframes_payload.get("duration", 0.0) or 0.0)
    candidates = sorted(
        keyframes_payload.get("candidates", []), key=lambda item: float(item["time"])
    )
    if not candidates:
        return [
            {
                "window_id": "video_1",
                "start_time": 0.0,
                "end_time": round(duration, 3),
                "frames": [],
                "is_cover": False,
            }
        ]

    boundaries = [0.0]
    for item in candidates:
        if item["time"] > boundaries[-1]:
            boundaries.append(round(float(item["time"]), 3))
    if boundaries[-1] < duration:
        boundaries.append(round(duration, 3))

    raw_windows: list[dict[str, Any]] = []
    for idx in range(len(boundaries) - 1):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        is_last_window = idx == len(boundaries) - 2
        frames = [
            item
            for item in candidates
            if start <= float(item["time"]) < end
            or (is_last_window and float(item["time"]) == end)
        ]
        raw_windows.append(
            {
                "window_id": f"video_{idx + 1}",
                "start_time": round(start, 3),
                "end_time": round(end, 3),
                "frames": frames,
                "is_cover": False,
                "max_global_score": max(
                    float(frame.get("global_score", 0.0))
                    for frame in frames
                    if float(frame.get("time", 0.0)) > float(start)
                )
                if any(float(frame.get("time", 0.0)) > float(start) for frame in frames)
                else 0.0,
            }
        )

    merged: list[dict[str, Any]] = []
    for window in raw_windows:
        duration_sec = float(window["end_time"]) - float(window["start_time"])
        merged_duration = (
            float(window["end_time"]) - float(merged[-1]["start_time"])
            if merged
            else duration_sec
        )
        if (
            merged
            and duration_sec < min_window_sec
            and float(window.get("max_global_score", 0.0)) < 12.0
            and merged_duration <= max_window_sec
        ):
            merged[-1]["end_time"] = window["end_time"]
            merged[-1]["frames"].extend(window.get("frames", []))
            merged[-1]["max_global_score"] = max(
                float(merged[-1].get("max_global_score", 0.0)),
                float(window.get("max_global_score", 0.0)),
            )
            continue
        merged.append(window)

    if len(merged) >= 2:
        last_duration = float(merged[-1]["end_time"]) - float(merged[-1]["start_time"])
        merged_duration = float(merged[-1]["end_time"]) - float(merged[-2]["start_time"])
        if (
            last_duration < min_window_sec
            and float(merged[-1].get("max_global_score", 0.0)) < 12.0
            and merged_duration <= max_window_sec
        ):
            merged[-2]["end_time"] = merged[-1]["end_time"]
            merged[-2]["frames"].extend(merged[-1].get("frames", []))
            merged[-2]["max_global_score"] = max(
                float(merged[-2].get("max_global_score", 0.0)),
                float(merged[-1].get("max_global_score", 0.0)),
            )
            merged.pop()

    for index, window in enumerate(merged, start=1):
        deduped_frames: list[dict[str, Any]] = []
        seen_pairs: set[tuple[float, str]] = set()
        for frame in window.get("frames", []):
            key = (round(float(frame["time"]), 3), str(frame["image_path"]))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            deduped_frames.append(frame)
        window["window_id"] = f"video_{index}"
        window["frames"] = deduped_frames
    return merged


def estimate_tts_duration(text: str, chars_per_sec: float = 4.0) -> float:
    clean = str(text or "").strip()
    if not clean:
        return 0.0
    return round(len(clean) / chars_per_sec, 3)


def calculate_window_max_chars(
    window: dict[str, Any], chars_per_keyframe: int = 50
) -> int:
    keyframe_count = max(1, len(window.get("frames", [])))
    return int(chars_per_keyframe * keyframe_count)


def compress_text_to_limit(text: str, limit: int) -> str:
    clean = " ".join(str(text or "").split()).strip()
    if len(clean) <= limit:
        return clean

    clauses = [item.strip() for item in re.split(r"[锛?銆傦紱;锛?]", clean) if item.strip()]
    for clause in reversed(clauses):
        if len(clause) <= limit:
            return clause

    trimmed = clean[:limit].rstrip(" 锛?銆傦紱;锛?")
    if trimmed and trimmed[-1].isascii() and trimmed[-1].isalpha():
        while trimmed and trimmed[-1].isascii() and trimmed[-1].isalpha():
            trimmed = trimmed[:-1]
        trimmed = trimmed.rstrip(" -_/")
    return trimmed or clean[:limit]


def sanitize_transition_prefix(text: str, previous_context_type: str) -> str:
    clean = " ".join(str(text or "").split()).strip()
    if not clean:
        return clean
    if previous_context_type not in {"start", "silent_gap"}:
        return clean

    prefixes = [
        "首先",
        "第一步",
        "接下来",
        "然后",
        "接着",
        "现在",
        "这里",
    ]
    changed = True
    while changed and clean:
        changed = False
        lower_clean = clean.lower()
        for prefix in prefixes:
            if lower_clean.startswith(prefix):
                clean = clean[len(prefix):].lstrip(" ，,。；;：:.-")
                changed = True
                break
    return clean


def build_batch_prompt(
    windows: list[dict[str, Any]],
    global_summary: str,
    previous_context: str,
    reference_text: str,
) -> str:
    lines = [
        "你正在为视频片段生成简洁的口播旁白。",
        "只返回 JSON 数组，每个对象包含 window_id、spoken_text、is_silent、reason 和 reference_terms_used。",
        "请结合全局摘要和上一段旁白，保持上下文连贯。",
        "如果有大号文字，重点关注画面中的大号文字，这类文字通常是对当前步骤或画面重点的注解。",
        "尽量让每个 spoken_text 不超过对应的字数预算。",
        f"全局摘要：{global_summary}",
        f"上一段上下文：{previous_context}",
    ]
    if previous_context == "[START_OF_VIDEO]":
        lines.append("这是视频的第一个口播窗口，请直接进入有用内容。")
    elif previous_context.startswith("[SILENT_GAP:"):
        lines.append("这一批窗口前有一段静默，请自然恢复讲解，不要特意说明静默。")
    if reference_text:
        lines.append(f"参考文本：{reference_text}")
    lines.append("窗口字数预算：")
    for window in windows:
        budget = int(window["max_chars"])
        lines.append(
            f"- {window['window_id']}: {float(window['start_time']):.2f}s - {float(window['end_time']):.2f}s，约 {budget} 个字"
        )
    lines.append(
        json.dumps(
            [
                {
                    "window_id": window["window_id"],
                    "spoken_text": "这个窗口对应的简洁旁白。",
                    "is_silent": False,
                    "reason": "画面有明显变化",
                    "reference_terms_used": [],
                }
                for window in windows
            ],
            ensure_ascii=False,
        )
    )
    return "\n".join(lines)


def _fallback_window_result(window: dict[str, Any], is_silent: bool) -> dict[str, Any]:
    if window.get("is_cover"):
        return {
            "window_id": window["window_id"],
            "spoken_text": "这里是本视频的开场背景。",
            "is_silent": False,
            "reason": "fallback_cover",
            "reference_terms_used": [],
        }
    if is_silent:
        return {
            "window_id": window["window_id"],
            "spoken_text": "",
            "is_silent": True,
            "reason": "low_visual_change",
            "reference_terms_used": [],
        }
    return {
        "window_id": window["window_id"],
        "spoken_text": "这个窗口展示了视频中的明显变化。",
        "is_silent": False,
        "reason": "fallback_text",
        "reference_terms_used": [],
    }


def _normalize_batch_results(
    windows: list[dict[str, Any]], results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result_map = {str(item.get("window_id")): item for item in results}
    normalized: list[dict[str, Any]] = []
    for window in windows:
        is_silent = not window.get("is_cover") and not _window_has_meaningful_change(window)
        item = result_map.get(window["window_id"])
        if item is None:
            item = _fallback_window_result(window, is_silent=is_silent)
        spoken_text = sanitize_transition_prefix(
            str(item.get("spoken_text", "")).strip(),
            str(window.get("previous_context_type", "")),
        )
        if is_silent and not spoken_text:
            item["is_silent"] = True
        item["window_id"] = window["window_id"]
        item["spoken_text"] = spoken_text
        item.setdefault("reference_terms_used", [])
        item["needs_review"] = bool(
            estimate_tts_duration(item["spoken_text"]) > float(window["duration_budget"])
        )
        normalized.append(item)
    return normalized


def generate_window_scripts(
    windows: list[dict[str, Any]],
    reference_text: str,
    global_summary: str,
    gemini_api_key: str | None,
    batch_size: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    drafts: list[dict[str, Any]] = []
    requests_log: list[dict[str, Any]] = []
    responses_log: list[dict[str, Any]] = []
    previous_item: dict[str, Any] | None = None
    resolved_api_key = load_gemini_api_key(gemini_api_key)

    batch_starts = range(0, len(windows), batch_size)
    for start in batch_starts:
        batch = windows[start : start + batch_size]
        previous_context, previous_context_type = build_previous_context(previous_item)
        prepared_batch: list[dict[str, Any]] = []
        batch_previous_type = previous_context_type
        for window in batch:
            prepared = dict(window)
            prepared["previous_context_type"] = batch_previous_type
            prepared_batch.append(prepared)
            batch_previous_type = "spoken_text"
        prompt = build_batch_prompt(
            prepared_batch,
            global_summary=global_summary,
            previous_context=previous_context,
            reference_text=reference_text,
        )
        requests_log.append(
            {
                "window_ids": [item["window_id"] for item in prepared_batch],
                "previous_context": previous_context,
                "previous_context_type": previous_context_type,
                "prompt": prompt,
            }
        )
        try:
            response = call_vl_gemini(
                api_key=resolved_api_key,
                windows=prepared_batch,
                prompt=prompt,
            )
            response_text = str(response.get("content", "")).strip()
            parsed = parse_gemini_batch_response(response_text)
        except Exception as exc:
            response_text = f"error: {exc}"
            parsed = []
        responses_log.append(
            {
                "window_ids": [item["window_id"] for item in prepared_batch],
                "raw_content": response_text,
                "parsed_count": len(parsed),
            }
        )
        normalized = _normalize_batch_results(prepared_batch, parsed)
        batch_previous_item = previous_item
        for window, item in zip(prepared_batch, normalized):
            is_silent = bool(item.get("is_silent", False))
            item_previous_context, item_previous_context_type = build_previous_context(
                batch_previous_item
            )
            draft = {
                "window_id": window["window_id"],
                "start_time": round(float(window["start_time"]), 3),
                "end_time": round(float(window["end_time"]), 3),
                "source_frames": [
                    {
                        "time": round(float(frame["time"]), 3),
                        "image_path": frame["image_path"],
                    }
                    for frame in window.get("frames", [])
                ],
                "spoken_text": "" if is_silent else str(item.get("spoken_text", "")).strip(),
                "is_silent": is_silent,
                "silent_duration": round(
                    float(window["end_time"]) - float(window["start_time"]), 3
                )
                if is_silent
                else 0.0,
                "is_cover": bool(window.get("is_cover", False)),
                "previous_context_used": item_previous_context,
                "previous_context_type": item_previous_context_type,
                "reference_terms_used": item.get("reference_terms_used", []),
                "reason": item.get("reason", ""),
                "duration_budget": round(float(window["duration_budget"]), 3),
                "max_chars": int(window["max_chars"]),
                "estimated_tts_duration": estimate_tts_duration(item.get("spoken_text", "")),
                "needs_review": bool(item.get("needs_review", False)),
            }
            drafts.append(draft)
            previous_item = draft
            batch_previous_item = draft
    return drafts, requests_log, responses_log


def generate_cover_intro(
    cover_window: dict[str, Any],
    reference_text: str,
    gemini_api_key: str | None,
) -> tuple[dict[str, Any], str]:
    resolved_api_key = load_gemini_api_key(gemini_api_key)
    prompt = build_cover_prompt(reference_text)
    try:
        response = call_vl_gemini(
            api_key=resolved_api_key,
            windows=[cover_window],
            prompt=prompt,
        )
        parsed = safe_parse_json_from_content(str(response.get("content", "")))
    except Exception:
        parsed = {}
    spoken_text = str(parsed.get("spoken_text", "")).strip() or "这里是本视频的开场背景。"
    draft = {
        "window_id": cover_window["window_id"],
        "start_time": round(float(cover_window["start_time"]), 3),
        "end_time": round(float(cover_window["end_time"]), 3),
        "source_frames": [
            {"time": 0.0, "image_path": cover_window["frames"][0]["image_path"]}
        ],
        "spoken_text": spoken_text,
        "is_silent": False,
        "silent_duration": 0.0,
        "is_cover": True,
        "previous_context_used": "[START_OF_VIDEO]",
        "previous_context_type": "start",
        "reference_terms_used": parsed.get("reference_terms_used", []),
        "reason": "cover_intro",
        "duration_budget": cover_window["duration_budget"],
        "max_chars": int(cover_window["max_chars"]),
        "estimated_tts_duration": estimate_tts_duration(spoken_text),
        "needs_review": False,
    }
    global_summary = str(parsed.get("global_summary", "")).strip()
    return draft, global_summary or build_global_summary(
        cover_image=Path(cover_window["frames"][0]["image_path"]),
        reference_text=reference_text,
    )


def build_spoken_payload(
    video: Path,
    drafts: list[dict[str, Any]],
    global_summary: str,
    reference_text_path: Path | None,
    ocr_enabled: bool,
) -> dict[str, Any]:
    paragraphs: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    cover_paragraph_index: int | None = None
    for index, draft in enumerate(drafts, start=1):
        normalized_spoken_text = sanitize_transition_prefix(
            draft.get("spoken_text", ""),
            str(draft.get("previous_context_type", "")),
        )
        paragraph = {
            "index": index,
            "is_title": False,
            "is_cover": bool(draft.get("is_cover", False)),
            "is_silent": bool(draft.get("is_silent", False)),
            "source_text": normalized_spoken_text,
            "spoken_text": normalized_spoken_text,
            "segment_count": 0 if draft.get("is_silent") else 1,
            "start_time": draft["start_time"],
            "end_time": draft["end_time"],
            "source_frames": draft.get("source_frames", []),
            "source_window_id": draft["window_id"],
            "duration_budget": draft["duration_budget"],
            "estimated_tts_duration": float(
                draft.get("estimated_tts_duration", estimate_tts_duration(normalized_spoken_text))
            ),
            "needs_review": bool(draft.get("needs_review", False)),
            "silent_duration": draft.get("silent_duration", 0.0),
            "previous_context_used": draft.get("previous_context_used", ""),
            "previous_context_type": draft.get("previous_context_type", ""),
            "reference_terms_used": draft.get("reference_terms_used", []),
        }
        paragraphs.append(paragraph)
        if paragraph["is_cover"]:
            cover_paragraph_index = index
        if not paragraph["is_silent"] and paragraph["spoken_text"]:
            segments.append(
                {
                    "segment_id": f"p01_s{index:03d}",
                    "paragraph_index": index,
                    "source_text": paragraph["spoken_text"],
                    "spoken_text": paragraph["spoken_text"],
                    "start_time": paragraph["start_time"],
                    "end_time": paragraph["end_time"],
                    "is_cover": paragraph["is_cover"],
                    "is_silent": False,
                }
            )
    return {
        "source_path": str(video.resolve()),
        "source_type": "video_context_windows",
        "video_path": str(video.resolve()),
        "page": 1,
        "title_text": normalize_video_title(video.stem),
        "global_summary": global_summary,
        "reference_text_path": str(reference_text_path.resolve()) if reference_text_path else None,
        "ocr_enabled": bool(ocr_enabled),
        "window_count": len(drafts),
        "batch_size": 3,
        "cover_paragraph_index": cover_paragraph_index,
        "paragraphs": paragraphs,
        "segments": segments,
        "combined_spoken_text": " ".join(
            item["spoken_text"] for item in paragraphs if item["spoken_text"]
        ),
    }


def run_video_script_generate(
    video: Path,
    output_dir: Path,
    debug_dir: Path,
    gemini_api_key: str | None = None,
    reference_text_path: Path | None = None,
    cover_image: Path | None = None,
    cover_duration_sec: float | None = None,
    enable_ocr: bool = False,
    batch_size: int = 3,
    frame_stride: int | None = None,
    min_gap_sec: float = 2.0,
    global_threshold: float = 12.0,
    subtitle_threshold: float = 8.0,
    detection_max_width: int = 960,
    fill_gap_sec: float = 6.0,
) -> dict[str, Any]:
    debug_dir.mkdir(parents=True, exist_ok=True)
    reference_text = read_reference_text(reference_text_path)
    windows: list[dict[str, Any]] = []
    if cover_image is not None:
        windows.append(
            build_cover_window(
                cover_image=cover_image,
                cover_duration_sec=cover_duration_sec or 3.0,
            )
        )
    keyframes_payload = sample_keyframes(
        video_path=video,
        out_dir=debug_dir / "keyframes",
        frame_stride=frame_stride,
        min_gap_sec=min_gap_sec,
        global_threshold=global_threshold,
        subtitle_threshold=subtitle_threshold,
        detection_max_width=detection_max_width,
        fill_gap_sec=fill_gap_sec,
    )
    write_json(debug_dir / "keyframes.json", keyframes_payload)
    windows.extend(build_video_windows(keyframes_payload))

    for window in windows:
        window_duration = max(
            0.5, float(window["end_time"]) - float(window["start_time"])
        )
        window["duration_budget"] = round(window_duration + 0.8, 3)
        window["max_chars"] = calculate_window_max_chars(window)

    write_json(debug_dir / "window_manifest.json", {"windows": windows})
    cover_draft: dict[str, Any] | None = None
    if windows and windows[0].get("is_cover"):
        cover_draft, cover_summary = generate_cover_intro(
            cover_window=windows[0],
            reference_text=reference_text,
            gemini_api_key=gemini_api_key,
        )
        global_summary = cover_summary
        windows = windows[1:]
    else:
        global_summary = build_global_summary(
            cover_image=cover_image,
            reference_text=reference_text,
        )
    write_json(debug_dir / "global_summary.json", {"global_summary": global_summary})

    drafts, request_logs, response_logs = generate_window_scripts(
        windows=windows,
        reference_text=reference_text,
        global_summary=global_summary,
        gemini_api_key=gemini_api_key,
        batch_size=batch_size,
    )
    if cover_draft is not None:
        drafts = [cover_draft] + drafts
    write_json(debug_dir / "window_script_drafts.json", {"drafts": drafts})
    write_json(debug_dir / "gemini_batch_requests.json", {"requests": request_logs})
    write_json(debug_dir / "gemini_batch_responses.json", {"responses": response_logs})

    spoken_payload = build_spoken_payload(
        video=video,
        drafts=drafts,
        global_summary=global_summary,
        reference_text_path=reference_text_path,
        ocr_enabled=enable_ocr,
    )
    spoken_path = output_dir / "page_01.spoken.json"
    extracted_path = output_dir / "page_01.extracted.json"
    write_json(spoken_path, spoken_payload)
    write_json(
        extracted_path,
        {
            "source_path": str(video.resolve()),
            "source_type": "video_context_windows",
            "page": 1,
            "paragraph_count": len(spoken_payload["paragraphs"]),
            "paragraphs": spoken_payload["paragraphs"],
            "global_summary": global_summary,
        },
    )
    return {
        "spoken_path": spoken_path,
        "extracted_path": extracted_path,
        "spoken": spoken_payload,
        "extracted": json.loads(extracted_path.read_text(encoding="utf-8")),
    }
