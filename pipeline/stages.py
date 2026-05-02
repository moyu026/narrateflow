from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pipeline.paths import infer_page_from_spoken_json, resolve_profile_path
from pipeline.runtime import reload_runtime_config
from pipeline.shared import video_output_root, write_json
from pipeline.ui import (
    ask_regenerate_target,
    ask_regenerate_volume_gain,
    ask_stage2_action,
    ask_voice_generation_scope,
    parse_paragraph_indices,
    prompt_choice,
    show_profile_summary,
    show_stage2_summary,
    stage_banner,
)
from timeline_align.run_timeline_align import run_timeline_align
from timeline_align.video_script import (
    generate_video_script_from_windows,
    prepare_video_script_windows,
)
from video_compose.run_video_compose import run_video_compose


def default_stage1_output_dir(config: dict[str, Any]) -> Path | None:
    if config.get("stage1_output_dir"):
        return Path(config["stage1_output_dir"])
    return video_output_root(config["video"]) / "scripts"


def default_timeline_output(spoken_json: Path, config: dict[str, Any]) -> Path:
    if config.get("timeline_output"):
        return Path(config["timeline_output"])
    page = int(config.get("page") or infer_page_from_spoken_json(spoken_json))
    return video_output_root(config["video"]) / "timeline" / f"page_{page:02d}.timeline.final.json"


def default_timeline_debug_dir(
    config: dict[str, Any], spoken_json: Path
) -> Path | None:
    if config.get("timeline_debug_dir"):
        return Path(config["timeline_debug_dir"])
    return video_output_root(config["video"]) / "timeline" / "debug"


def default_compose_output_dir(source_path: Path, config: dict[str, Any]) -> Path:
    if config.get("compose_output_dir"):
        return Path(config["compose_output_dir"])
    return video_output_root(config["video"]) / "composed"


def run_stage1(config: dict[str, Any]) -> dict[str, Any]:
    prepared = run_stage1_keyframes(config)
    return run_stage1_vlm(config, prepared)


def run_stage1_keyframes(config: dict[str, Any]) -> dict[str, Any]:
    debug_dir = (
        Path(config["timeline_debug_dir"])
        if config.get("timeline_debug_dir")
        else video_output_root(config["video"]) / "timeline_debug"
    )
    return prepare_video_script_windows(
        video=Path(config["video"]),
        debug_dir=debug_dir,
        cover_image=Path(config["cover_image"]) if config.get("cover_image") else None,
        cover_duration_sec=config.get("cover_duration_sec"),
        frame_stride=config.get("frame_stride"),
        min_gap_sec=float(config.get("min_gap_sec") or 2.0),
        global_threshold=float(config.get("global_threshold") or 12.0),
        subtitle_threshold=float(config.get("subtitle_threshold") or 8.0),
        detection_max_width=int(config.get("detection_max_width") or 960),
        fill_gap_sec=float(config.get("fill_gap_sec") or 6.0),
    )


def run_stage1_vlm(config: dict[str, Any], prepared: dict[str, Any]) -> dict[str, Any]:
    output_dir = default_stage1_output_dir(config)
    debug_dir = (
        Path(config["timeline_debug_dir"])
        if config.get("timeline_debug_dir")
        else video_output_root(config["video"]) / "timeline_debug"
    )
    return generate_video_script_from_windows(
        video=Path(config["video"]),
        output_dir=output_dir,
        debug_dir=debug_dir,
        windows=prepared["windows"],
        gemini_api_key=config.get("api_key"),
        reference_text_path=Path(config["reference_text"]) if config.get("reference_text") else None,
        enable_ocr=bool(config.get("enable_ocr", False)),
    )


def run_stage2_profile(config: dict[str, Any]) -> Path:
    from voice_process.run_voice_profile import run_voice_profile

    return run_voice_profile(
        voice_name=config["voice_name"],
        ref_audio=config["ref_audio"],
        ref_text=config["ref_text"],
        output_dir=Path(config["profile_output_dir"])
        if config.get("profile_output_dir")
        else None,
    )


def run_stage2_voice(
    config: dict[str, Any],
    spoken_json: Path,
    profile_path: Path,
    paragraph_indices: list[int] | None = None,
    volume_gain: float | None = None,
) -> Path:
    from voice_process.run_voice_generate import run_voice_generate

    if not paragraph_indices:
        result = run_voice_generate(
            spoken_json=spoken_json,
            profile_path=profile_path,
            voice_name=config.get("voice_name") or profile_path.stem,
            volume_gain=volume_gain,
            output_dir=Path(config["voice_output_dir"])
            if config.get("voice_output_dir")
            else video_output_root(config["video"]) / "voice",
        )
        return Path(result["manifest_path"])

    manifest_path: Path | None = None
    for paragraph_index in paragraph_indices:
        result = run_voice_generate(
            spoken_json=spoken_json,
            profile_path=profile_path,
            voice_name=config.get("voice_name") or profile_path.stem,
            paragraph_index=paragraph_index,
            volume_gain=volume_gain,
            output_dir=Path(config["voice_output_dir"])
            if config.get("voice_output_dir")
            else video_output_root(config["video"]) / "voice",
        )
        manifest_path = Path(result["manifest_path"])
    if manifest_path is None:
        raise ValueError("No valid paragraph indices provided for generation.")
    return manifest_path


def rerun_stage2_for_target(
    config: dict[str, Any],
    spoken_json: Path,
    profile_path: Path,
    target: str,
    volume_gain: float | None = None,
) -> Path:
    if target.strip().lower() == "all":
        return run_stage2_voice(
            config,
            spoken_json,
            profile_path,
            paragraph_indices=None,
            volume_gain=volume_gain,
        )
    return run_stage2_voice(
        config,
        spoken_json,
        profile_path,
        paragraph_indices=parse_paragraph_indices(target),
        volume_gain=volume_gain,
    )


def ask_only_voice_action(*, allow_back: bool) -> str:
    return prompt_choice(
        "Voice generation review action\n- r: regenerate one or more paragraphs\n- s: stop here\nChoice",
        ["r", "s"],
        default="s",
    )


def ensure_profile_ready(
    config: dict[str, Any], total: int
) -> tuple[dict[str, Any], Path]:
    profile_path = Path(config["profile"]) if config.get("profile") else None
    if profile_path is None:
        stage_banner(3, total, "Voice Profile Generation")
        profile_path = run_stage2_profile(config)
        config["profile"] = str(profile_path)
        show_profile_summary(profile_path)
        return config, profile_path

    print()
    print(f"[3/{total}] Voice Profile Generation (skipped, using existing profile)")
    print(f"profile_path: {profile_path}")
    return config, profile_path


def run_voice_stage_with_review(
    args: argparse.Namespace,
    config: dict[str, Any],
    spoken_json: Path,
    profile_path: Path,
    allow_back: bool,
    prompt_if_missing: bool,
    action_selector,
) -> tuple[dict[str, Any], Path, str]:
    config = reload_runtime_config(args, config)
    paragraph_indices, volume_gain = ask_voice_generation_scope(
        config,
        prompt_if_missing=prompt_if_missing,
    )
    manifest_path = run_stage2_voice(
        config,
        spoken_json,
        profile_path,
        paragraph_indices=paragraph_indices,
        volume_gain=volume_gain,
    )
    while True:
        show_stage2_summary(manifest_path)
        action = action_selector(allow_back=allow_back)
        if action in {"c", "s", "b"}:
            return config, manifest_path, action
        target = ask_regenerate_target()
        volume_gain = ask_regenerate_volume_gain()
        config = reload_runtime_config(args, config)
        manifest_path = rerun_stage2_for_target(
            config,
            spoken_json,
            profile_path,
            target,
            volume_gain=volume_gain,
        )


def run_profile_and_voice_stages(
    args: argparse.Namespace,
    config: dict[str, Any],
    total: int,
    spoken_json: Path,
    allow_back: bool,
    prompt_if_missing: bool,
) -> tuple[dict[str, Any], Path, str]:
    config = reload_runtime_config(args, config)
    config, profile_path = ensure_profile_ready(config, total)
    stage_banner(4, total, "Voice Generation")
    return run_voice_stage_with_review(
        args=args,
        config=config,
        spoken_json=spoken_json,
        profile_path=profile_path,
        allow_back=allow_back,
        prompt_if_missing=prompt_if_missing,
        action_selector=ask_stage2_action,
    )


def run_stage3(config: dict[str, Any], spoken_json: Path) -> Path:
    output = default_timeline_output(spoken_json, config)
    run_timeline_align(
        video=Path(config["video"]),
        spoken_json=spoken_json,
        output=output,
        debug_dir=default_timeline_debug_dir(config, spoken_json),
        api_key=config["api_key"],
        cover_paragraph_index=(
            int(config.get("cover_paragraph_index") or 2)
            if config.get("cover_image")
            else None
        ),
    )
    return output


def resolve_outro_profile_path(config: dict[str, Any]) -> Path:
    if config.get("outro_profile"):
        return resolve_profile_path(config["outro_profile"])
    if config.get("profile"):
        return Path(config["profile"])
    raise ValueError("Voice profile is required to generate outro slogan audio.")


def generate_outro_audio(config: dict[str, Any], output_dir: Path) -> Path | None:
    if not config.get("outro_image"):
        return None
    if config.get("outro_audio"):
        return Path(config["outro_audio"])
    if not config.get("outro_text"):
        return None

    import soundfile as sf

    from voice_process.common import load_model, load_prompt_file, synthesize_segment_wavs

    profile_path = resolve_outro_profile_path(config)
    prompt_items = load_prompt_file(profile_path)
    tts = load_model()
    segments = [
        {
            "segment_id": "outro_slogan",
            "paragraph_index": 9999,
            "spoken_text": config["outro_text"],
        }
    ]
    wavs, sample_rate = synthesize_segment_wavs(
        tts=tts,
        prompt_items=prompt_items,
        segments=segments,
        language="Chinese",
        speed=1.0,
        max_new_tokens=1024,
        batch_size=1,
    )
    outro_dir = output_dir / "outro"
    outro_dir.mkdir(parents=True, exist_ok=True)
    outro_audio_path = outro_dir / "outro_slogan.wav"
    outro_meta_path = outro_dir / "outro_slogan.json"
    sf.write(outro_audio_path, wavs[0], sample_rate)
    write_json(
        outro_meta_path,
        {
            "profile_path": str(profile_path),
            "text": config["outro_text"],
            "wav_path": str(outro_audio_path),
            "sample_rate": sample_rate,
            "duration": round(float(len(wavs[0]) / sample_rate), 3),
        },
    )
    return outro_audio_path


def run_stage4(
    config: dict[str, Any], timeline_path: Path, manifest_path: Path
) -> Path:
    output_dir = default_compose_output_dir(timeline_path, config)
    outro_audio = generate_outro_audio(config, output_dir)
    return run_video_compose(
        video=Path(config["video"]),
        timeline=timeline_path,
        segments_manifest=manifest_path,
        output_dir=output_dir,
        cover_image=Path(config["cover_image"]) if config.get("cover_image") else None,
        cover_duration_sec=config.get("cover_duration_sec"),
        cover_paragraph_index=int(config.get("cover_paragraph_index") or 2),
        outro_image=Path(config["outro_image"]) if config.get("outro_image") else None,
        outro_audio=outro_audio,
    )
