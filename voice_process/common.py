from __future__ import annotations

from datetime import datetime
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voice_clone_tool import (  # noqa: E402
    apply_fade,
    apply_speed,
    load_model,
    load_prompt_file,
    match_segment_levels,
    save_prompt_file,
    segment_pause_ms,
    slugify,
    write_json,
)


OUTPUTS_DIR = ROOT / "outputs"


def build_voice_output_dir(profile_name: str, page_title: str) -> Path:
    out_dir = OUTPUTS_DIR / slugify(profile_name) / slugify(page_title, max_len=60)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_voice_file_stem(ppt_path: str | None, page: int) -> str:
    ppt_stem = Path(ppt_path).stem if ppt_path else f"page_{page:02d}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}__{slugify(f'{ppt_stem}_page_{page:02d}', max_len=36)}"


def synthesize_segment_wavs(
    tts,
    prompt_items,
    segments: list[dict],
    language: str,
    speed: float,
    max_new_tokens: int,
    batch_size: int = 4,
) -> tuple[list[np.ndarray], int]:
    if not segments:
        raise ValueError("没有可合成的片段。")

    adjusted: list[np.ndarray] = []
    sample_rate: int | None = None
    batch_size = max(1, int(batch_size))

    for start in range(0, len(segments), batch_size):
        batch = segments[start : start + batch_size]
        texts = [segment["spoken_text"] for segment in batch]
        batch_prompt_items = (
            prompt_items * len(batch)
            if len(prompt_items) == 1
            else prompt_items[start : start + len(batch)]
        )
        wavs, batch_sample_rate = tts.generate_voice_clone(
            text=texts,
            language=[language] * len(texts),
            voice_clone_prompt=batch_prompt_items,
            max_new_tokens=max_new_tokens,
        )
        if sample_rate is None:
            sample_rate = batch_sample_rate
        adjusted.extend(
            apply_speed(np.asarray(wav, dtype=np.float32), speed=speed) for wav in wavs
        )
        if hasattr(__import__("torch"), "cuda"):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if sample_rate is None:
        raise RuntimeError("分段合成失败，未返回采样率。")

    adjusted = match_segment_levels(adjusted)
    adjusted = [apply_fade(wav, sample_rate=sample_rate) for wav in adjusted]
    return adjusted, sample_rate


def write_segment_outputs(
    out_dir: Path,
    segments: list[dict],
    wavs: list[np.ndarray],
    sample_rate: int,
    pause_ms: int,
) -> list[dict]:
    segment_dir = out_dir / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    manifest_segments: list[dict] = []
    cursor = 0.0
    for index, (segment, wav) in enumerate(zip(segments, wavs)):
        paragraph_index = int(segment["paragraph_index"])
        segment_id = segment["segment_id"]
        segment_path = segment_dir / f"p{paragraph_index:02d}__{segment_id}.wav"
        sf.write(segment_path, wav, sample_rate)
        duration = float(len(wav) / sample_rate)
        start = round(cursor, 3)
        end = round(cursor + duration, 3)
        manifest_segments.append(
            {
                "paragraph_index": paragraph_index,
                "segment_id": segment_id,
                "wav_path": str(segment_path),
                "start": start,
                "end": end,
                "duration": round(duration, 3),
                "spoken_text": segment["spoken_text"],
            }
        )
        cursor = end
        if index < len(segments) - 1:
            cursor += float(segment_pause_ms(segment["spoken_text"], pause_ms) / 1000)
    return manifest_segments


def recalculate_manifest_timings(
    manifest_segments: list[dict], pause_ms: int
) -> list[dict]:
    normalized: list[dict] = []
    cursor = 0.0
    ordered = sorted(
        manifest_segments,
        key=lambda item: (int(item["paragraph_index"]), str(item["segment_id"])),
    )
    for index, item in enumerate(ordered):
        duration = round(float(item["duration"]), 3)
        start = round(cursor, 3)
        end = round(cursor + duration, 3)
        normalized.append(
            {
                **item,
                "start": start,
                "end": end,
                "duration": duration,
            }
        )
        cursor = end
        if index < len(ordered) - 1:
            cursor += float(segment_pause_ms(item["spoken_text"], pause_ms) / 1000)
    return normalized


__all__ = [
    "ROOT",
    "OUTPUTS_DIR",
    "build_voice_output_dir",
    "build_voice_file_stem",
    "load_model",
    "load_prompt_file",
    "save_prompt_file",
    "slugify",
    "recalculate_manifest_timings",
    "synthesize_segment_wavs",
    "write_segment_outputs",
    "write_json",
]
