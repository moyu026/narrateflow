from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def get_video_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def get_video_resolution(video_path: Path) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    width_text, height_text = result.stdout.strip().split("x", 1)
    return int(width_text), int(height_text)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_timeline_segments(timeline_path: Path) -> list[dict]:
    payload = load_json(timeline_path)
    return payload.get("segments") or payload.get("timeline") or []


def resolve_end_hint(item: dict, fallback: float) -> float:
    value = item.get("end_hint", item.get("anchor_end"))
    if value is None:
        return float(fallback)
    return float(value)


def build_direct_track(
    timeline_path: Path,
    manifest_path: Path,
    video_duration: float,
    buffer_sec: float = 0.8,
) -> tuple[np.ndarray, int, list[dict]]:
    timeline = load_timeline_segments(timeline_path)
    manifest = load_json(manifest_path)
    segment_map = {int(item["paragraph_index"]): item for item in manifest["segments"]}
    sample_rate = int(manifest["sample_rate"])
    matched = [
        item
        for item in timeline
        if item.get("matched")
        and not item.get("is_cover")
        and int(item["paragraph_index"]) in segment_map
    ]
    matched.sort(
        key=lambda item: (
            float(item.get("start", item.get("anchor_start"))),
            int(item["paragraph_index"]),
        )
    )
    total_samples = int(np.ceil(video_duration * sample_rate))
    track = np.zeros(total_samples, dtype=np.float32)
    placements = []
    for index, item in enumerate(matched):
        idx = int(item["paragraph_index"])
        segment = segment_map[idx]
        wav, sr = sf.read(segment["wav_path"], dtype="float32")
        if int(sr) != sample_rate:
            raise RuntimeError("segment sample rate mismatch")
        start = float(item.get("start", item.get("anchor_start")))
        next_start = (
            float(
                matched[index + 1].get("start", matched[index + 1].get("anchor_start"))
            )
            if index + 1 < len(matched)
            else video_duration
        )
        required_duration = float(len(wav) / sample_rate) + buffer_sec
        available_duration = max(0.0, next_start - start)
        end = min(video_duration, start + float(len(wav) / sample_rate))
        start_sample = int(round(start * sample_rate))
        if len(wav) > 0:
            track[start_sample : start_sample + len(wav)] += wav
        placements.append(
            {
                "paragraph_index": idx,
                "spoken_text": item["spoken_text"],
                "start": round(start, 3),
                "end_hint": round(resolve_end_hint(item, start), 3),
                "placed_start": round(start, 3),
                "placed_end": round(end, 3),
                "audio_duration": round(float(len(wav) / sample_rate), 3),
                "buffer_sec": round(buffer_sec, 3),
                "required_duration": round(required_duration, 3),
                "available_duration": round(available_duration, 3),
                "needs_retime": bool(
                    available_duration < required_duration and index + 1 < len(matched)
                ),
                "segment_wav": segment["wav_path"],
            }
        )
    return np.clip(track, -0.98, 0.98), sample_rate, placements


def build_retime_segments(
    timeline_path: Path,
    manifest_path: Path,
    video_duration: float,
    buffer_sec: float = 0.8,
    tail_buffer_sec: float = 1.0,
) -> list[dict]:
    timeline = load_timeline_segments(timeline_path)
    manifest = load_json(manifest_path)
    segment_map = {int(item["paragraph_index"]): item for item in manifest["segments"]}
    matched = [
        item
        for item in timeline
        if item.get("matched") and not item.get("is_cover")
    ]
    matched.sort(key=lambda item: float(item.get("start", item.get("anchor_start"))))
    segments = []
    for i, item in enumerate(matched):
        start = float(item.get("start", item.get("anchor_start")))
        next_start = resolve_end_hint(item, video_duration)
        source_duration = max(0.01, next_start - start)
        audio_meta = segment_map.get(int(item["paragraph_index"]))
        audio_duration = float(audio_meta["duration"]) if audio_meta is not None else 0.0
        is_silent = bool(item.get("is_silent")) or audio_meta is None
        effective_buffer = tail_buffer_sec if i + 1 == len(matched) else buffer_sec
        desired_duration = (
            source_duration
            if is_silent
            else max(source_duration, audio_duration + effective_buffer)
        )
        retime_factor = desired_duration / source_duration if source_duration else 1.0
        output_duration = source_duration * retime_factor
        segments.append(
            {
                "paragraph_index": int(item["paragraph_index"]),
                "spoken_text": item["spoken_text"],
                "start": round(start, 3),
                "end_hint": round(resolve_end_hint(item, next_start), 3),
                "source_start": round(start, 3),
                "source_end": round(next_start, 3),
                "source_duration": round(source_duration, 3),
                "audio_duration": round(audio_duration, 3),
                "buffer_sec": round(effective_buffer, 3),
                "desired_duration": round(desired_duration, 3),
                "output_duration": round(output_duration, 3),
                "retime_factor": round(retime_factor, 4),
                "needs_review": bool(audio_duration > output_duration + 0.05),
                "segment_wav": audio_meta["wav_path"] if audio_meta is not None else None,
                "is_silent": is_silent,
            }
        )
    return segments


def build_retime_track(
    segments: list[dict], sample_rate: int, audio_tail_pad_sec: float = 0.5
) -> tuple[np.ndarray, list[dict]]:
    parts = []
    placements = []
    cursor = 0.0
    for segment in segments:
        segment_samples = int(round(float(segment["output_duration"]) * sample_rate))
        chunk = np.zeros(segment_samples, dtype=np.float32)
        if segment.get("segment_wav"):
            wav, sr = sf.read(segment["segment_wav"], dtype="float32")
            if int(sr) != sample_rate:
                raise RuntimeError("segment sample rate mismatch")
            wav = np.asarray(wav, dtype=np.float32)
            copy_len = min(len(wav), len(chunk))
            chunk[:copy_len] = wav[:copy_len]
        parts.append(chunk)
        placements.append(
            {
                "paragraph_index": segment["paragraph_index"],
                "placed_start": round(cursor, 3),
                "placed_end": round(cursor + float(segment["output_duration"]), 3),
                "audio_duration": segment["audio_duration"],
                "source_duration": segment["source_duration"],
                "desired_duration": segment["desired_duration"],
                "retime_factor": segment["retime_factor"],
                "needs_review": segment["needs_review"],
                "segment_wav": segment["segment_wav"],
                "is_silent": bool(segment.get("is_silent", False)),
            }
        )
        cursor += float(segment["output_duration"])
    if audio_tail_pad_sec > 0:
        parts.append(
            np.zeros(int(round(audio_tail_pad_sec * sample_rate)), dtype=np.float32)
        )
    return (
        np.concatenate(parts).astype(np.float32)
        if parts
        else np.zeros(1, dtype=np.float32)
    ), placements


def render_retimed_video(
    video_path: Path, segments: list[dict], output_video: Path
) -> None:
    filter_parts = []
    concat_inputs = []
    for idx, segment in enumerate(segments):
        filter_parts.append(
            f"[0:v]trim=start={segment['source_start']}:end={segment['source_end']},setpts={segment['retime_factor']}*(PTS-STARTPTS),fps=30,format=yuv420p[v{idx}]"
        )
        concat_inputs.append(f"[v{idx}]")
    filter_complex = ";".join(
        filter_parts
        + ["".join(concat_inputs) + f"concat=n={len(segments)}:v=1:a=0[vout]"]
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-an",
            "-r",
            "30",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            str(output_video),
        ],
        check=True,
    )


def mux_video(video_path: Path, audio_path: Path, output_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            str(output_path),
        ],
        check=True,
    )


def build_cover_video(
    cover_image: Path,
    duration_sec: float,
    output_video: Path,
    width: int,
    height: int,
) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(cover_image),
            "-t",
            f"{duration_sec}",
            "-vf",
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,fps=30,format=yuv420p",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            str(output_video),
        ],
        check=True,
    )


def prepend_cover_to_video(
    video_path: Path,
    cover_image: Path,
    cover_duration_sec: float,
    output_video: Path,
    work_dir: Path,
    width: int,
    height: int,
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    cover_video = work_dir / "cover_intro.mp4"
    concat_list = work_dir / "video_concat.txt"
    build_cover_video(cover_image, cover_duration_sec, cover_video, width, height)
    concat_list.write_text(
        f"file '{cover_video.resolve().as_posix()}'\nfile '{video_path.resolve().as_posix()}'\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(output_video),
        ],
        check=True,
    )
    return output_video


def infer_cover_duration_sec(
    manifest: dict,
    requested_duration_sec: float | None = None,
    paragraph_index: int = 2,
) -> float:
    if requested_duration_sec is not None:
        return float(requested_duration_sec)
    segments = manifest.get("segments", [])
    if not segments:
        raise ValueError("Cannot infer cover duration without manifest segments.")
    for item in segments:
        if int(item["paragraph_index"]) == paragraph_index:
            return float(item["duration"])
    raise ValueError(
        f"Cannot infer cover duration: paragraph_index={paragraph_index} not found in manifest."
    )


def build_cover_audio_prefix(
    manifest: dict,
    paragraph_index: int,
    sample_rate: int,
    cover_duration_sec: float,
) -> tuple[np.ndarray, dict]:
    segments = manifest.get("segments", [])
    cover_segment = next(
        (item for item in segments if int(item["paragraph_index"]) == paragraph_index),
        None,
    )
    if cover_segment is None:
        raise ValueError(
            f"Cover paragraph_index={paragraph_index} not found in manifest."
        )
    wav, sr = sf.read(cover_segment["wav_path"], dtype="float32")
    if int(sr) != sample_rate:
        raise RuntimeError("cover segment sample rate mismatch")
    wav = np.asarray(wav, dtype=np.float32)
    total_samples = int(round(float(cover_duration_sec) * sample_rate))
    if total_samples < len(wav):
        raise ValueError(
            "cover duration is shorter than the selected cover paragraph audio."
        )
    prefix = np.zeros(max(total_samples, len(wav)), dtype=np.float32)
    prefix[: len(wav)] += wav
    placement = {
        "paragraph_index": int(paragraph_index),
        "placed_start": 0.0,
        "placed_end": round(float(len(wav) / sample_rate), 3),
        "cover_duration_sec": round(float(cover_duration_sec), 3),
        "audio_duration": round(float(len(wav) / sample_rate), 3),
        "segment_wav": cover_segment["wav_path"],
        "is_cover_intro": True,
    }
    return prefix, placement


def get_audio_duration(audio_path: Path) -> float:
    info = sf.info(audio_path)
    return float(info.frames / info.samplerate)


def append_outro_to_video(
    video_path: Path,
    outro_image: Path,
    outro_duration_sec: float,
    output_video: Path,
    work_dir: Path,
    width: int,
    height: int,
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    outro_video = work_dir / "outro.mp4"
    concat_list = work_dir / "outro_concat.txt"
    build_cover_video(outro_image, outro_duration_sec, outro_video, width, height)
    concat_list.write_text(
        f"file '{video_path.resolve().as_posix()}'\nfile '{outro_video.resolve().as_posix()}'\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(output_video),
        ],
        check=True,
    )
    return output_video


def build_outro_audio_suffix(
    outro_audio: Path,
    sample_rate: int,
    placed_start: float,
) -> tuple[np.ndarray, dict]:
    wav, sr = sf.read(outro_audio, dtype="float32")
    wav = np.asarray(wav, dtype=np.float32)
    if int(sr) != sample_rate:
        wav = librosa.resample(wav, orig_sr=int(sr), target_sr=sample_rate)
        wav = np.asarray(wav, dtype=np.float32)
    duration = float(len(wav) / sample_rate)
    placement = {
        "placed_start": round(placed_start, 3),
        "placed_end": round(placed_start + duration, 3),
        "audio_duration": round(duration, 3),
        "segment_wav": str(outro_audio),
        "is_outro": True,
    }
    return wav, placement


def run_video_compose(
    video: Path,
    timeline: Path,
    segments_manifest: Path,
    output_dir: Path,
    buffer_sec: float = 1.2,
    tail_buffer_sec: float = 1.5,
    audio_tail_pad_sec: float = 0.5,
    cover_image: Path | None = None,
    cover_duration_sec: float | None = None,
    cover_paragraph_index: int = 2,
    outro_image: Path | None = None,
    outro_audio: Path | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    video_duration = get_video_duration(video)
    manifest = load_json(segments_manifest)
    sample_rate = int(manifest["sample_rate"])
    video_width, video_height = get_video_resolution(video)

    segments = build_retime_segments(
        timeline,
        segments_manifest,
        video_duration,
        buffer_sec=buffer_sec,
        tail_buffer_sec=tail_buffer_sec,
    )
    audio_track, placements = build_retime_track(
        segments, sample_rate, audio_tail_pad_sec=audio_tail_pad_sec
    )
    render_retimed_video(video, segments, output_dir / "page_retimed_video.mp4")
    final_video_input = output_dir / "page_retimed_video.mp4"
    applied_cover_duration = None
    if cover_image is not None:
        applied_cover_duration = infer_cover_duration_sec(
            manifest,
            requested_duration_sec=cover_duration_sec,
            paragraph_index=cover_paragraph_index,
        )
        final_video_input = prepend_cover_to_video(
            video_path=final_video_input,
            cover_image=cover_image,
            cover_duration_sec=applied_cover_duration,
            output_video=output_dir / "page_retimed_video_with_cover.mp4",
            work_dir=output_dir / "cover_tmp",
            width=video_width,
            height=video_height,
        )
        cover_prefix, cover_placement = build_cover_audio_prefix(
            manifest=manifest,
            paragraph_index=cover_paragraph_index,
            sample_rate=sample_rate,
            cover_duration_sec=applied_cover_duration,
        )
        audio_track = np.concatenate([cover_prefix, audio_track]).astype(np.float32)
        shifted_placements = []
        for item in placements:
            shifted = dict(item)
            shifted["placed_start"] = round(
                float(item["placed_start"]) + applied_cover_duration, 3
            )
            shifted["placed_end"] = round(
                float(item["placed_end"]) + applied_cover_duration, 3
            )
            shifted_placements.append(shifted)
        placements = [cover_placement] + shifted_placements
    applied_outro_duration = None
    if outro_image is not None and outro_audio is not None:
        applied_outro_duration = get_audio_duration(outro_audio)
        final_video_input = append_outro_to_video(
            video_path=final_video_input,
            outro_image=outro_image,
            outro_duration_sec=applied_outro_duration,
            output_video=output_dir / "page_retimed_video_with_outro.mp4",
            work_dir=output_dir / "outro_tmp",
            width=video_width,
            height=video_height,
        )
        outro_start = float(len(audio_track) / sample_rate)
        outro_suffix, outro_placement = build_outro_audio_suffix(
            outro_audio=outro_audio,
            sample_rate=sample_rate,
            placed_start=outro_start,
        )
        audio_track = np.concatenate([audio_track, outro_suffix]).astype(np.float32)
        placements = placements + [outro_placement]
    sf.write(output_dir / "page_audio.wav", audio_track, sample_rate)
    mux_video(
        final_video_input,
        output_dir / "page_audio.wav",
        output_dir / "page_composed.mp4",
    )
    plan = {
        "mode": "retime",
        "video_path": str(video),
        "timeline": str(timeline),
        "segments_manifest": str(segments_manifest),
        "buffer_sec": buffer_sec,
        "tail_buffer_sec": tail_buffer_sec,
        "audio_tail_pad_sec": audio_tail_pad_sec,
        "cover_image": str(cover_image) if cover_image else None,
        "cover_duration_sec": applied_cover_duration,
        "cover_paragraph_index": cover_paragraph_index if cover_image else None,
        "outro_image": str(outro_image) if outro_image else None,
        "outro_audio": str(outro_audio) if outro_audio else None,
        "outro_duration_sec": applied_outro_duration,
        "segments": segments,
        "placements": placements,
    }

    (output_dir / "page_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_dir / "page_composed.mp4"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 video composition")
    parser.add_argument("--video", required=True)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--segments-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--buffer-sec", type=float, default=1.2)
    parser.add_argument("--tail-buffer-sec", type=float, default=1.5)
    parser.add_argument("--audio-tail-pad-sec", type=float, default=0.5)
    parser.add_argument("--cover-image", default=None)
    parser.add_argument("--cover-duration-sec", type=float, default=None)
    parser.add_argument("--cover-paragraph-index", type=int, default=2)
    parser.add_argument("--outro-image", default=None)
    parser.add_argument("--outro-audio", default=None)
    args = parser.parse_args()

    output = run_video_compose(
        video=Path(args.video),
        timeline=Path(args.timeline),
        segments_manifest=Path(args.segments_manifest),
        output_dir=Path(args.output_dir),
        buffer_sec=args.buffer_sec,
        tail_buffer_sec=args.tail_buffer_sec,
        audio_tail_pad_sec=args.audio_tail_pad_sec,
        cover_image=Path(args.cover_image) if args.cover_image else None,
        cover_duration_sec=args.cover_duration_sec,
        cover_paragraph_index=args.cover_paragraph_index,
        outro_image=Path(args.outro_image) if args.outro_image else None,
        outro_audio=Path(args.outro_audio) if args.outro_audio else None,
    )
    print(output)


if __name__ == "__main__":
    main()
