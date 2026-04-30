from __future__ import annotations

from datetime import datetime
from dataclasses import asdict
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOX_DIR = ROOT / "tools" / "sox" / "sox-14.4.2"
MODELS_DIR = ROOT / "models"

if SOX_DIR.exists():
    os.environ["PATH"] = str(SOX_DIR) + os.pathsep + os.environ.get("PATH", "")

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem  # noqa: E402


OUTPUTS_DIR = ROOT / "outputs"


def slugify(value: str, max_len: int = 48) -> str:
    value = re.sub(r"\s+", "_", value.strip())
    value = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "", value)
    value = value.strip("._-")
    if not value:
        value = "untitled"
    return value[:max_len]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def detect_model_dir() -> Path:
    candidates = sorted(MODELS_DIR.glob("Qwen/Qwen3-TTS-12Hz-1*Base"))
    if not candidates:
        raise FileNotFoundError(
            "找不到本地模型目录，请确认模型已下载到 models/Qwen/..."
        )
    return candidates[0]


def recommended_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    capability = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if capability[0] >= 8 else torch.float16


def recommended_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def build_load_kwargs(
    device: str | None = None, dtype: str | None = None
) -> dict[str, Any]:
    device = device or recommended_device()
    dtype_obj = getattr(torch, dtype) if dtype else recommended_dtype()
    kwargs: dict[str, Any] = {
        "device_map": device,
        "dtype": dtype_obj,
        "low_cpu_mem_usage": True,
    }
    if device.startswith("cuda"):
        kwargs["attn_implementation"] = "sdpa"
    return kwargs


def load_model(
    model_dir: Path | None = None, device: str | None = None, dtype: str | None = None
) -> Qwen3TTSModel:
    model_dir = model_dir or detect_model_dir()
    return Qwen3TTSModel.from_pretrained(
        str(model_dir), **build_load_kwargs(device=device, dtype=dtype)
    )


def save_prompt_file(
    prompt_items: list[VoiceClonePromptItem], profile_path: Path
) -> None:
    payload = {"items": [asdict(item) for item in prompt_items]}
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, profile_path)


def load_prompt_file(profile_path: Path) -> list[VoiceClonePromptItem]:
    payload = torch.load(profile_path, map_location="cpu", weights_only=True)
    items_raw = payload["items"]
    items: list[VoiceClonePromptItem] = []
    for item in items_raw:
        ref_code = item.get("ref_code")
        if ref_code is not None and not torch.is_tensor(ref_code):
            ref_code = torch.tensor(ref_code)
        ref_spk = item["ref_spk_embedding"]
        if not torch.is_tensor(ref_spk):
            ref_spk = torch.tensor(ref_spk)
        items.append(
            VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=ref_spk,
                x_vector_only_mode=bool(item.get("x_vector_only_mode", False)),
                icl_mode=bool(
                    item.get(
                        "icl_mode", not bool(item.get("x_vector_only_mode", False))
                    )
                ),
                ref_text=item.get("ref_text"),
            )
        )
    return items


def segment_pause_ms(text: str, default_pause_ms: int) -> int:
    text = text.strip()
    if not text:
        return default_pause_ms
    if re.search(r"[。！？!?]$", text):
        return int(default_pause_ms * 1.4)
    if re.search(r"[；;：:]$", text):
        return int(default_pause_ms * 1.15)
    return default_pause_ms


def rms_level(wav: np.ndarray) -> float:
    wav = np.asarray(wav, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(wav), dtype=np.float64)) + 1e-8)


def match_segment_levels(
    wavs: list[np.ndarray], target_rms: float = 0.09
) -> list[np.ndarray]:
    if not wavs:
        return []
    matched: list[np.ndarray] = []
    prev_rms: float | None = None
    for wav in wavs:
        wav = np.asarray(wav, dtype=np.float32)
        current_rms = rms_level(wav)
        desired_rms = (
            target_rms if prev_rms is None else min(max(prev_rms, 0.055), 0.12)
        )
        gain = desired_rms / current_rms
        gain = min(max(gain, 0.85), 1.15)
        adjusted = np.clip(wav * gain, -0.98, 0.98).astype(np.float32)
        matched.append(adjusted)
        prev_rms = rms_level(adjusted)
    return matched


def apply_fade(wav: np.ndarray, sample_rate: int, fade_ms: int = 28) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).copy()
    fade_samples = min(int(sample_rate * fade_ms / 1000), len(wav) // 8)
    if fade_samples <= 1:
        return wav
    ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    wav[:fade_samples] *= ramp
    wav[-fade_samples:] *= ramp[::-1]
    return wav


def apply_volume_gain(wav: np.ndarray, gain: float) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if gain <= 0:
        raise ValueError("volume gain 必须大于 0。")
    if abs(gain - 1.0) < 1e-6:
        return wav
    return np.clip(wav * gain, -0.98, 0.98).astype(np.float32)


def apply_speed(wav: np.ndarray, speed: float) -> np.ndarray:
    if abs(speed - 1.0) < 1e-6:
        return wav.astype(np.float32)
    if speed <= 0:
        raise ValueError("speed 必须大于 0。")

    if SOX_DIR.exists():
        tmp_dir = ROOT / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        in_path = tmp_dir / f"speed_in_{ts}.wav"
        out_path = tmp_dir / f"speed_out_{ts}.wav"
        wav = np.asarray(wav, dtype=np.float32)
        sf.write(in_path, wav, 24000)
        cmd = f'"{SOX_DIR / "sox.exe"}" "{in_path}" "{out_path}" tempo {speed}'
        exit_code = os.system(cmd)
        if exit_code == 0 and out_path.exists():
            slowed, _ = sf.read(out_path, dtype="float32")
            try:
                in_path.unlink(missing_ok=True)
                out_path.unlink(missing_ok=True)
            except OSError:
                pass
            return np.asarray(slowed, dtype=np.float32)

    stretched = librosa.effects.time_stretch(
        np.asarray(wav, dtype=np.float32), rate=speed
    )
    return np.asarray(stretched, dtype=np.float32)


def build_voice_output_dir(
    profile_name: str, page_title: str, output_dir: Path | None = None
) -> Path:
    out_dir = (
        output_dir
        if output_dir is not None
        else OUTPUTS_DIR / slugify(profile_name) / slugify(page_title, max_len=60)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_voice_file_stem(source_path: str | None, page: int) -> str:
    source_stem = Path(source_path).stem if source_path else f"page_{page:02d}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}__{slugify(f'{source_stem}_page_{page:02d}', max_len=36)}"


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
    progress = tqdm(
        total=len(segments),
        desc="Generating voice",
        unit="segment",
        leave=False,
    )

    try:
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
            progress.update(len(batch))
            if hasattr(__import__("torch"), "cuda"):
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        progress.close()

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
    "apply_volume_gain",
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
