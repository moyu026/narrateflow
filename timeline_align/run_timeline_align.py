from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_voice_timeline_items(
    spoken_payload: dict[str, Any],
    cover_paragraph_index: int | None = None,
) -> list[dict[str, Any]]:
    body_start_paragraph_index = (
        int(cover_paragraph_index) + 1 if cover_paragraph_index is not None else None
    )
    timeline: list[dict[str, Any]] = []

    for paragraph in spoken_payload.get("paragraphs", []):
        paragraph_index = int(paragraph["index"])
        is_title = bool(paragraph.get("is_title", False))
        timeline_enabled = not is_title
        if (
            body_start_paragraph_index is not None
            and paragraph_index < body_start_paragraph_index
            and not is_title
        ):
            timeline_enabled = False

        item: dict[str, Any] = {
            "paragraph_index": paragraph_index,
            "role": "title" if is_title else "voice",
            "spoken_text": paragraph.get("spoken_text", ""),
            "timeline_enabled": timeline_enabled,
            "matched": False,
        }
        if not is_title and not timeline_enabled:
            item["review_status"] = "cover_intro"
        if paragraph.get("timeline_start_sec") is not None:
            item["anchor_start"] = round(float(paragraph["timeline_start_sec"]), 3)
            item["matched"] = timeline_enabled
        timeline.append(item)

    return timeline


def validate_timestamp_timeline(timeline: list[dict[str, Any]]) -> None:
    missing = [
        int(item["paragraph_index"])
        for item in timeline
        if item.get("role") == "voice"
        and item.get("timeline_enabled", True)
        and item.get("anchor_start") is None
    ]
    if missing:
        raise ValueError(
            "时间戳模式要求每个正文段落都带开始时间戳，缺少段落: "
            + ",".join(str(item) for item in missing)
        )

    previous_start: float | None = None
    previous_index: int | None = None
    for item in timeline:
        if item.get("role") != "voice" or not item.get("timeline_enabled", True):
            continue
        current_start = float(item["anchor_start"])
        current_index = int(item["paragraph_index"])
        if previous_start is not None and current_start < previous_start:
            raise ValueError(
                "时间戳必须按正文段落顺序非递减: "
                f"paragraph_index={previous_index} start={previous_start}, "
                f"paragraph_index={current_index} start={current_start}"
            )
        previous_start = current_start
        previous_index = current_index


def add_end_hints(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    voice_items = [
        item
        for item in timeline
        if item.get("role") == "voice"
        and item.get("timeline_enabled", True)
        and item.get("matched")
    ]
    for index, item in enumerate(voice_items):
        next_start = (
            float(voice_items[index + 1]["anchor_start"])
            if index + 1 < len(voice_items)
            else float(item["anchor_start"])
        )
        item["anchor_end"] = round(next_start, 3)
    return timeline


def build_timeline_status(timeline: list[dict[str, Any]]) -> dict[str, Any]:
    voice_items = [
        item
        for item in timeline
        if item.get("role") == "voice" and item.get("timeline_enabled", True)
    ]
    matched_items = [item for item in voice_items if item.get("matched")]
    missing_indices = [
        int(item["paragraph_index"]) for item in voice_items if not item.get("matched")
    ]
    return {
        "status": "complete" if len(voice_items) == len(matched_items) else "incomplete",
        "voice_count": len(voice_items),
        "matched_count": len(matched_items),
        "missing_paragraph_indices": missing_indices,
    }


def build_public_timeline(final_payload: dict[str, Any], spoken_json: Path) -> dict[str, Any]:
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
                if item.get("matched") and item.get("anchor_end") is not None
                else None,
                "review_status": item.get("review_status")
                or ("explicit_timestamp" if item.get("matched") else "needs_manual"),
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
            "时间戳模式：start 直接来自脚本文字段落开头的时间戳。",
            "end_hint 仅为参考结束点，不作为音频裁剪硬边界。",
            "最后一个正文段落的 end_hint 等于自身 start。",
        ],
    }


def build_timestamp_timeline(
    spoken_json: Path,
    cover_paragraph_index: int | None = None,
) -> dict[str, Any]:
    spoken_payload = load_json(spoken_json)
    timeline = build_voice_timeline_items(
        spoken_payload, cover_paragraph_index=cover_paragraph_index
    )
    validate_timestamp_timeline(timeline)
    timeline = add_end_hints(timeline)
    status = build_timeline_status(timeline)
    return {
        "spoken_json": str(spoken_json),
        "page": spoken_payload.get("page", 1),
        "mode": "script_timestamps",
        "timeline": timeline,
        "notes": [
            "该时间轴直接使用脚本文字时间戳，不进行关键帧采样。",
            "所有参与正文合成的段落都必须提供 timeline_start_sec。",
        ],
        **status,
    }


def run_timeline_align(
    video: Path,
    spoken_json: Path,
    output: Path,
    debug_dir: Path | None = None,
    cover_paragraph_index: int | None = None,
) -> dict[str, Any]:
    _ = video
    _ = debug_dir
    final_payload = build_timestamp_timeline(
        spoken_json=spoken_json,
        cover_paragraph_index=cover_paragraph_index,
    )
    debug_output = output.with_suffix(output.suffix + ".debug.json")
    public_payload = build_public_timeline(final_payload, spoken_json=spoken_json)
    write_json(debug_output, final_payload)
    write_json(output, public_payload)
    return public_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 timestamp timeline generation")
    parser.add_argument("--video", required=True)
    parser.add_argument("--spoken-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--debug-dir", default=None)
    parser.add_argument("--cover-paragraph-index", type=int, default=None)
    args = parser.parse_args()

    result = run_timeline_align(
        video=Path(args.video),
        spoken_json=Path(args.spoken_json),
        output=Path(args.output),
        debug_dir=Path(args.debug_dir) if args.debug_dir else None,
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
