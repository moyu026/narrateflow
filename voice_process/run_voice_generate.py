from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voice_process.common import (
    apply_volume_gain,
    build_voice_file_stem,
    build_voice_output_dir,
    load_model,
    load_prompt_file,
    recalculate_manifest_timings,
    synthesize_segment_wavs,
    write_segment_outputs,
    write_json,
)


def load_spoken_payload(spoken_json: Path) -> dict[str, Any]:
    if not spoken_json.exists():
        raise FileNotFoundError(f"spoken json not found: {spoken_json}")
    payload = json.loads(spoken_json.read_text(encoding="utf-8"))
    if not isinstance(payload.get("paragraphs"), list):
        raise ValueError("spoken json 缺少 paragraphs 字段，无法进入声音生成阶段。")
    return payload


def build_generation_units(payload: dict[str, Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    page = int(payload.get("page", 1))
    for paragraph in payload.get("paragraphs", []):
        if paragraph.get("is_title"):
            continue
        paragraph_index = int(paragraph["index"])
        spoken_text = str(paragraph.get("spoken_text", "")).strip()
        source_text = str(paragraph.get("source_text", "")).strip()
        if not spoken_text:
            continue
        units.append(
            {
                "segment_id": f"p{page:02d}_s{paragraph_index:03d}",
                "paragraph_index": paragraph_index,
                "source_text": source_text,
                "spoken_text": spoken_text,
            }
        )
    return units


def validate_profile(profile_path: Path) -> None:
    if profile_path.exists():
        return
    raise FileNotFoundError(
        "未找到指定音色文件。可先通过 voice_process/run_voice_profile.py 从参考音频生成 .pt 音色文件。"
        f"\nprofile path: {profile_path}"
    )


def run_voice_generate(
    spoken_json: Path,
    profile_path: Path,
    voice_name: str | None = None,
    language: str = "Chinese",
    pause_ms: int = 420,
    speed: float = 1.0,
    max_new_tokens: int = 1024,
    device: str | None = None,
    dtype: str | None = None,
    paragraph_index: int | None = None,
    segment_id: str | None = None,
    volume_gain: float | None = None,
    export_full_page: bool = False,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    payload = load_spoken_payload(spoken_json)
    validate_profile(profile_path)
    if export_full_page and (paragraph_index is not None or segment_id is not None):
        raise ValueError("单段重生成时不支持导出整页 wav，请移除 --export-full-page。")

    segments = build_generation_units(payload)
    if paragraph_index is not None:
        segments = [
            item for item in segments if int(item["paragraph_index"]) == paragraph_index
        ]
    if segment_id is not None:
        segments = [item for item in segments if str(item["segment_id"]) == segment_id]
    if not segments:
        raise ValueError("未找到要生成的段落，请检查 paragraph_index 或 segment_id。")

    prompt_items = load_prompt_file(profile_path)
    tts = load_model(device=device, dtype=dtype)
    wavs, sample_rate = synthesize_segment_wavs(
        tts=tts,
        prompt_items=prompt_items,
        segments=segments,
        language=language,
        speed=speed,
        max_new_tokens=max_new_tokens,
    )
    if volume_gain is not None:
        wavs = [apply_volume_gain(wav, volume_gain) for wav in wavs]

    resolved_voice_name = voice_name or profile_path.stem
    page = int(payload.get("page", 1))
    source_path_text = payload.get("source_path")
    title_text = payload.get("title_text") or f"page_{page:02d}"
    out_dir = build_voice_output_dir(
        profile_name=resolved_voice_name,
        page_title=title_text,
        output_dir=output_dir,
    )
    manifest_path = out_dir / "segments_manifest.json"

    existing_manifest = None
    if manifest_path.exists() and (
        paragraph_index is not None or segment_id is not None
    ):
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            existing_manifest = None

    new_manifest_segments = write_segment_outputs(
        out_dir=out_dir,
        segments=segments,
        wavs=wavs,
        sample_rate=sample_rate,
        pause_ms=pause_ms,
    )

    if existing_manifest and (paragraph_index is not None or segment_id is not None):
        segment_map = {
            item["segment_id"]: item for item in existing_manifest.get("segments", [])
        }
        for item in new_manifest_segments:
            segment_map[item["segment_id"]] = item
        merged_manifest_segments = recalculate_manifest_timings(
            list(segment_map.values()), pause_ms=pause_ms
        )
    else:
        merged_manifest_segments = recalculate_manifest_timings(
            new_manifest_segments, pause_ms=pause_ms
        )

    manifest_payload = {
        "page": page,
        "title_text": title_text,
        "sample_rate": sample_rate,
        "segments": merged_manifest_segments,
    }
    if (
        existing_manifest
        and existing_manifest.get("full_wav_path")
        and not export_full_page
    ):
        manifest_payload["full_wav_path"] = existing_manifest["full_wav_path"]
    if export_full_page:
        stem = build_voice_file_stem(source_path=source_path_text, page=page)
        full_wav_path = out_dir / f"{stem}.wav"
        full_meta_path = out_dir / f"{stem}.json"
        import numpy as np
        import soundfile as sf

        full_parts = []
        for index, wav in enumerate(wavs):
            full_parts.append(wav)
            if index < len(wavs) - 1:
                silence = np.zeros(int(sample_rate * pause_ms / 1000), dtype=np.float32)
                full_parts.append(silence)
        merged = (
            np.concatenate(full_parts).astype(np.float32)
            if full_parts
            else np.zeros(1, dtype=np.float32)
        )
        sf.write(full_wav_path, merged, sample_rate)
        write_json(
            full_meta_path,
            {
                "voice_name": resolved_voice_name,
                "language": language,
                "source_path": source_path_text,
                "page": page,
                "title_text": title_text,
                "profile_path": str(profile_path.resolve()),
                "spoken_path": str(spoken_json.resolve()),
                "pause_ms": pause_ms,
                "speed": speed,
                "sample_rate": sample_rate,
                "segment_count": len(merged_manifest_segments),
                "segments": merged_manifest_segments,
                "wav_path": str(full_wav_path),
            },
        )
        manifest_payload["full_wav_path"] = str(full_wav_path)

    write_json(manifest_path, manifest_payload)
    return {
        "manifest_path": manifest_path,
        "metadata": manifest_payload,
        "output_dir": out_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 voice generation")
    parser.add_argument("--spoken-json", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--voice-name", default=None)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--pause-ms", type=int, default=420)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--paragraph-index", type=int, default=None)
    parser.add_argument("--segment-id", default=None)
    parser.add_argument("--volume-gain", type=float, default=None)
    parser.add_argument("--export-full-page", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    result = run_voice_generate(
        spoken_json=Path(args.spoken_json),
        profile_path=Path(args.profile),
        voice_name=args.voice_name,
        language=args.language,
        pause_ms=args.pause_ms,
        speed=args.speed,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
        paragraph_index=args.paragraph_index,
        segment_id=args.segment_id,
        volume_gain=args.volume_gain,
        export_full_page=args.export_full_page,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result["output_dir"]),
                "manifest_path": str(result["manifest_path"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
