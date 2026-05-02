from __future__ import annotations

import argparse
import os
import tomllib
from pathlib import Path
from typing import Any

from pipeline.paths import (
    resolve_profile_path,
    validate_existing_file,
    validate_existing_path,
)
from pipeline.shared import (
    ROOT,
    RUN_MODE_CHOICES,
    STAGE_ALIASES,
    STAGE_SELECTION_CHOICES,
)
from pipeline.ui import prompt_choice, prompt_existing_path, prompt_text


def read_env_key(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    env_path = ROOT / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip()
    return None


def needs_script_inputs(run_mode: str, target_stage: str | None) -> bool:
    return run_mode == "full" or (run_mode == "only" and target_stage == "script")


def needs_video_input(run_mode: str, target_stage: str | None) -> bool:
    if run_mode == "full":
        return True
    return target_stage in {"timeline", "compose"} or (
        run_mode == "from" and target_stage in {"profile", "voice"}
    )


def needs_cover_options(run_mode: str, target_stage: str | None) -> bool:
    if run_mode == "full":
        return True
    if target_stage in {"timeline", "compose"}:
        return True
    return run_mode == "from" and target_stage in {"profile", "voice"}


def needs_outro_options(run_mode: str, target_stage: str | None) -> bool:
    if run_mode == "full":
        return True
    if target_stage == "compose":
        return True
    return run_mode == "from" and target_stage in {"profile", "voice", "timeline"}


def needs_profile_creation_inputs(
    run_mode: str, config: dict[str, Any], target_stage: str | None
) -> bool:
    if target_stage == "profile":
        return True
    if run_mode == "from" and target_stage == "profile":
        return True
    return False


def empty_to_none(value: Any) -> Any:
    if isinstance(value, str) and not value.strip():
        return None
    return value


def load_toml_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def apply_pipeline_mode_config(args: argparse.Namespace) -> str | None:
    args.config_mode = "video"
    args.skip_initial_input_review = True
    args.skip_optional_prompts = True
    args.config_path = args.video_config
    apply_video_mode_config(args, Path(args.video_config))
    return "video"


def reload_runtime_config(
    args: argparse.Namespace, current_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    if current_config is not None:
        sync_config_to_args(args, current_config)
    run_mode = getattr(args, "run_mode", None)
    target_stage = getattr(args, "target_stage", None)
    if getattr(args, "config_mode", None) == "video":
        config_path = Path(getattr(args, "config_path", None) or args.video_config)
        apply_video_mode_config(args, config_path)
        args.run_mode = run_mode
        args.target_stage = target_stage
    config = resolve_initial_args(args)
    sync_config_to_args(args, config)
    return config


def apply_video_mode_config(args: argparse.Namespace, config_path: Path) -> None:
    payload = load_toml_config(config_path)
    pipeline = payload.get("pipeline", {})
    input_config = payload.get("input", {})
    voice = payload.get("voice", {})
    timeline = payload.get("timeline", {})
    outputs = payload.get("outputs", {})
    cover = payload.get("cover", {})
    outro = payload.get("outro", {})
    generation = payload.get("generation", {})

    run_mode = pipeline.get("run_mode", "full")
    target_stage = empty_to_none(pipeline.get("target_stage"))
    if run_mode not in RUN_MODE_CHOICES:
        raise ValueError(f"Invalid video config pipeline.run_mode: {run_mode}")
    if target_stage and target_stage not in STAGE_SELECTION_CHOICES:
        raise ValueError(f"Invalid video config pipeline.target_stage: {target_stage}")
    args.only_stage = target_stage if run_mode == "only" else None
    args.from_stage = target_stage if run_mode == "from" else None
    args.config_run_mode = run_mode
    args.config_target_stage = target_stage

    args.video = empty_to_none(input_config.get("video")) or args.video
    if not args.video:
        raise ValueError("Video mode requires input.video in config/video_mode.toml")
    args.reference_text = empty_to_none(input_config.get("reference_document"))
    args.enable_ocr = bool(input_config.get("enable_ocr", False))

    args.profile = empty_to_none(voice.get("profile")) or args.profile
    args.voice_name = empty_to_none(voice.get("voice_name")) or args.voice_name
    args.ref_audio = empty_to_none(voice.get("ref_audio")) or args.ref_audio
    args.ref_text = empty_to_none(voice.get("ref_text")) or args.ref_text
    if run_mode == "full" and not args.profile and not (
        args.voice_name and args.ref_audio and args.ref_text
    ):
        raise ValueError(
            "Video mode full run requires either voice.profile or voice_name/ref_audio/ref_text in config/video_mode.toml"
        )

    frame_stride = empty_to_none(timeline.get("frame_stride"))
    if frame_stride is not None:
        args.frame_stride = int(frame_stride)
    min_gap_sec = empty_to_none(timeline.get("min_gap_sec"))
    if min_gap_sec is not None:
        args.min_gap_sec = float(min_gap_sec)
    global_threshold = empty_to_none(timeline.get("global_threshold"))
    if global_threshold is not None:
        args.global_threshold = float(global_threshold)
    subtitle_threshold = empty_to_none(timeline.get("subtitle_threshold"))
    if subtitle_threshold is not None:
        args.subtitle_threshold = float(subtitle_threshold)
    detection_max_width = empty_to_none(timeline.get("detection_max_width"))
    if detection_max_width is not None:
        args.detection_max_width = int(detection_max_width)
    fill_gap_sec = empty_to_none(timeline.get("fill_gap_sec"))
    if fill_gap_sec is not None:
        args.fill_gap_sec = float(fill_gap_sec)
    args.api_key = empty_to_none(timeline.get("api_key")) or args.api_key

    args.stage1_output_dir = empty_to_none(outputs.get("stage1_output_dir"))
    args.profile_output_dir = empty_to_none(outputs.get("profile_output_dir"))
    args.voice_output_dir = empty_to_none(outputs.get("voice_output_dir"))
    args.timeline_output = empty_to_none(outputs.get("timeline_output"))
    args.timeline_debug_dir = empty_to_none(outputs.get("timeline_debug_dir"))
    args.compose_output_dir = empty_to_none(outputs.get("compose_output_dir"))

    if bool(cover.get("enabled", False)):
        args.cover_image = empty_to_none(cover.get("image"))
        if not args.cover_image:
            raise ValueError("Video config cover.enabled requires cover.image")
        args.cover_duration_sec = empty_to_none(cover.get("duration_sec"))
        args.cover_paragraph_index = int(cover.get("paragraph_index", 1))
    else:
        args.cover_image = None
        args.cover_duration_sec = None
        args.cover_paragraph_index = 2

    if bool(outro.get("enabled", False)):
        args.outro_image = empty_to_none(outro.get("image"))
        args.outro_audio = empty_to_none(outro.get("audio"))
        args.outro_text = empty_to_none(outro.get("text"))
        args.outro_profile = empty_to_none(outro.get("profile"))
        if not args.outro_image:
            raise ValueError("Video config outro.enabled requires outro.image")
        if not args.outro_audio and not args.outro_text:
            raise ValueError("Video config outro.enabled requires outro.audio or outro.text")
    else:
        args.outro_image = None
        args.outro_audio = None
        args.outro_text = None
        args.outro_profile = None

    args.paragraphs = empty_to_none(generation.get("paragraphs"))
    args.volume_gain = empty_to_none(generation.get("volume_gain"))


def resolve_initial_args(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    run_mode = args.run_mode
    target_stage = args.target_stage

    config["page"] = args.page
    config["video"] = None
    config["spoken_json"] = None
    config["timeline"] = None
    config["segments_manifest"] = None
    config["outro_profile"] = None
    config["reference_text"] = getattr(args, "reference_text", None)
    config["enable_ocr"] = bool(getattr(args, "enable_ocr", False))
    config["frame_stride"] = getattr(args, "frame_stride", None)
    config["min_gap_sec"] = float(getattr(args, "min_gap_sec", 2.0) or 2.0)
    config["global_threshold"] = float(
        getattr(args, "global_threshold", 12.0) or 12.0
    )
    config["subtitle_threshold"] = float(
        getattr(args, "subtitle_threshold", 8.0) or 8.0
    )
    config["detection_max_width"] = int(
        getattr(args, "detection_max_width", 960) or 960
    )
    config["fill_gap_sec"] = float(getattr(args, "fill_gap_sec", 6.0) or 6.0)

    if needs_script_inputs(run_mode, target_stage):
        config["page"] = 1
    elif target_stage in {"profile", "voice", "timeline"} and run_mode == "from":
        config["spoken_json"] = (
            validate_existing_file(args.spoken_json, "spoken json")
            if args.spoken_json
            else prompt_existing_path("Spoken JSON path")
        )
        if target_stage == "timeline":
            config["segments_manifest"] = (
                validate_existing_file(args.segments_manifest, "segments manifest")
                if args.segments_manifest
                else prompt_existing_path("Segments manifest path")
            )
    elif target_stage in {"voice", "timeline"}:
        config["spoken_json"] = (
            validate_existing_file(args.spoken_json, "spoken json")
            if args.spoken_json
            else prompt_existing_path("Spoken JSON path")
        )
    elif target_stage == "compose":
        config["timeline"] = (
            validate_existing_file(args.timeline, "timeline json")
            if args.timeline
            else prompt_existing_path("Timeline JSON path")
        )
        config["segments_manifest"] = (
            validate_existing_file(args.segments_manifest, "segments manifest")
            if args.segments_manifest
            else prompt_existing_path("Segments manifest path")
        )

    if needs_video_input(run_mode, target_stage):
        config["video"] = (
            validate_existing_file(args.video, "video path")
            if args.video
            else prompt_existing_path("Target video path")
        )

    profile = args.profile
    voice_name = args.voice_name
    ref_audio = args.ref_audio
    ref_text = args.ref_text

    if target_stage == "voice":
        if not profile:
            profile = prompt_existing_path(
                "Voice profile path (.pt file or profile directory)",
                kinds=("file", "dir"),
            )
    elif needs_profile_creation_inputs(run_mode, config, target_stage):
        voice_name = voice_name or prompt_text("Voice name")
        ref_audio = (
            validate_existing_file(ref_audio, "reference audio path")
            if ref_audio
            else prompt_existing_path(
                "Reference audio path (recommended 10-20 seconds)"
            )
        )
        ref_text = ref_text or prompt_text("Reference text")
    elif run_mode == "full":
        if not profile and not (voice_name and ref_audio and ref_text):
            has_profile = prompt_choice(
                "Do you already have a voice profile file", ["y", "n"], default="y"
            )
            if has_profile == "y":
                profile = prompt_existing_path(
                    "Voice profile path (.pt file or profile directory)",
                    kinds=("file", "dir"),
                )
            else:
                voice_name = voice_name or prompt_text("Voice name")
                ref_audio = ref_audio or prompt_existing_path(
                    "Reference audio path (recommended 10-20 seconds)"
                )
                ref_text = ref_text or prompt_text("Reference text")

    config["profile"] = str(resolve_profile_path(profile)) if profile else None
    config["voice_name"] = voice_name
    config["ref_audio"] = (
        validate_existing_file(ref_audio, "reference audio path") if ref_audio else None
    )
    config["ref_text"] = ref_text

    config["stage1_output_dir"] = args.stage1_output_dir
    config["profile_output_dir"] = args.profile_output_dir
    config["voice_output_dir"] = args.voice_output_dir
    config["timeline_output"] = args.timeline_output
    config["timeline_debug_dir"] = args.timeline_debug_dir
    config["compose_output_dir"] = args.compose_output_dir
    cover_image = (
        validate_existing_file(args.cover_image, "cover image") if args.cover_image else None
    )
    cover_duration_sec = args.cover_duration_sec
    cover_paragraph_index = args.cover_paragraph_index
    if (
        cover_image is None
        and needs_cover_options(run_mode, target_stage)
        and not getattr(args, "skip_optional_prompts", False)
    ):
        use_cover = prompt_choice(
            "Do you want to prepend a cover image before the main video",
            ["y", "n"],
            default="n",
        )
        if use_cover == "y":
            cover_image = prompt_existing_path("Cover image path")
            cover_paragraph_index = int(
                prompt_text(
                    "Cover paragraph index",
                    default=str(cover_paragraph_index or 2),
                )
            )
            if cover_duration_sec is None:
                raw_cover_duration = prompt_text(
                    "Optional cover duration in seconds (empty means use cover paragraph audio duration)",
                    default="",
                    required=False,
                ).strip()
                cover_duration_sec = float(raw_cover_duration) if raw_cover_duration else None
        else:
            cover_paragraph_index = None
            cover_duration_sec = None
    elif cover_image is not None:
        cover_paragraph_index = 1
    config["cover_image"] = cover_image
    config["cover_duration_sec"] = cover_duration_sec
    config["cover_paragraph_index"] = cover_paragraph_index
    outro_image = (
        validate_existing_file(args.outro_image, "outro image") if args.outro_image else None
    )
    outro_audio = (
        validate_existing_file(args.outro_audio, "outro audio") if args.outro_audio else None
    )
    outro_text = args.outro_text
    outro_profile = (
        validate_existing_path(args.outro_profile, "outro profile", kinds=("file", "dir"))
        if args.outro_profile
        else None
    )
    if (
        outro_image is not None
        and outro_audio is None
        and not outro_text
        and not getattr(args, "skip_optional_prompts", False)
    ):
        has_fixed_outro_audio = prompt_choice(
            "Do you already have a fixed outro slogan audio",
            ["y", "n"],
            default="y",
        )
        if has_fixed_outro_audio == "y":
            outro_audio = prompt_existing_path("Outro audio path")
        else:
            outro_text = prompt_text("Outro slogan text")
            if run_mode in {"only", "from"} and target_stage in {"compose", "timeline"} and not config.get("profile"):
                outro_profile = prompt_existing_path(
                    "Voice profile path for outro generation (.pt file or profile directory)",
                    kinds=("file", "dir"),
                )
    if (
        outro_image is None
        and needs_outro_options(run_mode, target_stage)
        and not getattr(args, "skip_optional_prompts", False)
    ):
        use_outro = prompt_choice(
            "Do you want to append an outro page after the main video",
            ["y", "n"],
            default="n",
        )
        if use_outro == "y":
            outro_image = prompt_existing_path("Outro image path")
            has_fixed_outro_audio = prompt_choice(
                "Do you already have a fixed outro slogan audio",
                ["y", "n"],
                default="y",
            )
            if has_fixed_outro_audio == "y":
                outro_audio = prompt_existing_path("Outro audio path")
            else:
                outro_text = outro_text or prompt_text("Outro slogan text")
                if run_mode in {"only", "from"} and target_stage in {"compose", "timeline"} and not config.get("profile"):
                    outro_profile = prompt_existing_path(
                        "Voice profile path for outro generation (.pt file or profile directory)",
                        kinds=("file", "dir"),
                    )
        else:
            outro_audio = None
            outro_text = None
            outro_profile = None
    config["outro_image"] = outro_image
    config["outro_audio"] = outro_audio
    config["outro_text"] = outro_text
    config["outro_profile"] = outro_profile
    config["paragraphs"] = args.paragraphs
    config["volume_gain"] = args.volume_gain
    env_key_name = "GEMINI_API_KEY"
    config["api_key"] = args.api_key or read_env_key(env_key_name)
    if not config["api_key"] and not getattr(args, "skip_optional_prompts", False):
        config["api_key"] = prompt_text(env_key_name, required=True)
    return config


def summarize_initial_inputs(
    config: dict[str, Any], run_mode: str, target_stage: str | None
) -> list[str]:
    lines = [f"run_mode: {run_mode}"]
    if target_stage is not None:
        lines.append(f"target_stage: {target_stage}")

    if needs_script_inputs(run_mode, target_stage):
        lines.append("script_source: video_context")
        lines.append(f"page: {config.get('page')}")

    if config.get("spoken_json"):
        lines.append(f"spoken_json: {config.get('spoken_json')}")
    if config.get("timeline"):
        lines.append(f"timeline: {config.get('timeline')}")
    if config.get("segments_manifest"):
        lines.append(f"segments_manifest: {config.get('segments_manifest')}")
    if config.get("video"):
        lines.append(f"video: {config.get('video')}")
    if config.get("reference_text"):
        lines.append(f"reference_text: {config.get('reference_text')}")
    if config.get("enable_ocr"):
        lines.append("enable_ocr: true")

    if config.get("profile"):
        lines.append(f"profile: {config.get('profile')}")
    elif config.get("voice_name") or config.get("ref_audio") or config.get("ref_text"):
        lines.append(f"voice_name: {config.get('voice_name')}")
        lines.append(f"ref_audio: {config.get('ref_audio')}")
        lines.append(f"ref_text: {config.get('ref_text')}")

    if config.get("cover_image"):
        lines.append(f"cover_image: {config.get('cover_image')}")
        lines.append(
            f"cover_paragraph_index: {config.get('cover_paragraph_index') or 2}"
        )
        lines.append(
            "cover_duration_sec: "
            + (
                str(config.get("cover_duration_sec"))
                if config.get("cover_duration_sec") is not None
                else "auto"
            )
        )

    if config.get("outro_image"):
        lines.append(f"outro_image: {config.get('outro_image')}")
        lines.append(
            "outro_audio: " + (config.get("outro_audio") or "generate from profile")
        )
        if config.get("outro_text"):
            lines.append(f"outro_text: {config.get('outro_text')}")
        if config.get("outro_profile"):
            lines.append(f"outro_profile: {config.get('outro_profile')}")

    if config.get("paragraphs"):
        lines.append(f"paragraphs: {config.get('paragraphs')}")
    if config.get("volume_gain") is not None:
        lines.append(f"volume_gain: {config.get('volume_gain')}")

    lines.append("api_key: " + ("set" if config.get("api_key") else "not set"))
    return lines


def confirm_initial_inputs(
    config: dict[str, Any], run_mode: str, target_stage: str | None
) -> str:
    print()
    print("Collected inputs:")
    for line in summarize_initial_inputs(config, run_mode, target_stage):
        print(f"- {line}")
    return prompt_choice(
        "Input action\n- c: continue\n- e: edit inputs for this run\n- s: stop here\nChoice",
        ["c", "e", "s"],
        default="c",
    )


def sync_config_to_args(args: argparse.Namespace, config: dict[str, Any]) -> None:
    args.page = config.get("page")
    args.video = config.get("video")
    args.spoken_json = config.get("spoken_json")
    args.timeline = config.get("timeline")
    args.segments_manifest = config.get("segments_manifest")
    args.profile = config.get("profile")
    args.voice_name = config.get("voice_name")
    args.ref_audio = config.get("ref_audio")
    args.ref_text = config.get("ref_text")
    args.reference_text = config.get("reference_text")
    args.enable_ocr = bool(config.get("enable_ocr", False))
    args.cover_image = config.get("cover_image")
    args.cover_duration_sec = config.get("cover_duration_sec")
    args.cover_paragraph_index = config.get("cover_paragraph_index")
    args.outro_image = config.get("outro_image")
    args.outro_audio = config.get("outro_audio")
    args.outro_text = config.get("outro_text")
    args.outro_profile = config.get("outro_profile")
    args.paragraphs = config.get("paragraphs")
    args.volume_gain = config.get("volume_gain")
    args.api_key = config.get("api_key")
    args.frame_stride = config.get("frame_stride")


def available_edit_sections(
    config: dict[str, Any], run_mode: str, target_stage: str | None
) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    if needs_script_inputs(run_mode, target_stage):
        sections.append(("script", "video script inputs"))
    if config.get("spoken_json") or config.get("timeline") or config.get("segments_manifest"):
        sections.append(("paths", "stage paths"))
    if config.get("video"):
        sections.append(("video", "video path"))
    if (
        config.get("profile")
        or config.get("voice_name")
        or config.get("ref_audio")
        or config.get("ref_text")
        or target_stage in {"profile", "voice"}
        or run_mode == "full"
    ):
        sections.append(("profile", "voice/profile inputs"))
    if needs_cover_options(run_mode, target_stage):
        sections.append(("cover", "cover intro"))
    if needs_outro_options(run_mode, target_stage):
        sections.append(("outro", "outro page"))
    if run_mode == "full" or target_stage in {"timeline"}:
        sections.append(("probe", "timeline probe"))
    seen: set[str] = set()
    unique_sections: list[tuple[str, str]] = []
    for key, label in sections:
        if key in seen:
            continue
        seen.add(key)
        unique_sections.append((key, label))
    return unique_sections


def ask_edit_section(
    config: dict[str, Any], run_mode: str, target_stage: str | None
) -> str:
    sections = available_edit_sections(config, run_mode, target_stage)
    if len(sections) == 1:
        return sections[0][0]
    print()
    print("Edit which section:")
    for key, label in sections:
        print(f"- {key}: {label}")
    return prompt_choice(
        "Section",
        [key for key, _ in sections],
        default=sections[0][0],
    )


def clear_args_for_section(args: argparse.Namespace, section: str) -> None:
    if section == "script":
        args.page = None
        args.reference_text = None
        args.enable_ocr = False
        args.frame_stride = None
        return
    if section == "paths":
        args.spoken_json = None
        args.timeline = None
        args.segments_manifest = None
        return
    if section == "video":
        args.video = None
        return
    if section == "profile":
        args.profile = None
        args.voice_name = None
        args.ref_audio = None
        args.ref_text = None
        return
    if section == "cover":
        args.cover_image = None
        args.cover_duration_sec = None
        args.cover_paragraph_index = 2
        return
    if section == "outro":
        args.outro_image = None
        args.outro_audio = None
        args.outro_text = None
        args.outro_profile = None
        return
    if section == "probe":
        args.api_key = None
        return


def resolve_run_plan(args: argparse.Namespace) -> tuple[str, str | None]:
    if args.only_stage and args.from_stage:
        raise ValueError("Use either --only-stage or --from-stage, not both.")

    if getattr(args, "config_run_mode", None):
        return args.config_run_mode, getattr(args, "config_target_stage", None)

    if args.only_stage:
        target_stage = STAGE_ALIASES.get(args.only_stage, args.only_stage)
        if target_stage in {"script", "profile", "voice", "timeline", "compose"}:
            return "only", target_stage
        raise ValueError(
            "Unsupported --only-stage value. Use one of: script, profile, voice, timeline, compose"
        )

    if args.from_stage:
        target_stage = STAGE_ALIASES.get(args.from_stage, args.from_stage)
        if target_stage in {"script", "profile", "voice", "timeline", "compose"}:
            return "from", target_stage
        raise ValueError(
            "Unsupported --from-stage value. Use one of: script, profile, voice, timeline, compose"
        )

    run_mode = prompt_choice(
        "Run mode\n- full: run the full pipeline\n- only: run only one stage\n- from: start from one stage and continue\nChoice",
        RUN_MODE_CHOICES,
        default="full",
    )
    if run_mode == "full":
        return "full", None
    target_stage = prompt_choice(
        "Choose stage",
        STAGE_SELECTION_CHOICES,
        default="voice" if run_mode == "from" else "script",
    )
    return run_mode, target_stage
