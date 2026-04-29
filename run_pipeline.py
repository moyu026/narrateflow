from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from text_process.run_text_process import prepare_ppt_page, slugify
from timeline_align.run_timeline_align import run_timeline_align
from timeline_align.video_script import run_video_script_generate
from video_compose.run_video_compose import run_video_compose


ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
STAGE_ALIASES = {
    "1": "text",
    "2": "profile",
    "3": "voice",
    "4": "timeline",
    "5": "compose",
    "text": "text",
    "profile": "profile",
    "voice": "voice",
    "timeline": "timeline",
    "compose": "compose",
}

STAGE_SELECTION_CHOICES = ["text", "profile", "voice", "timeline", "compose"]
RUN_MODE_CHOICES = ["full", "only", "from"]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def prompt_text(label: str, default: str | None = None, required: bool = True) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"{label}{suffix}: ").strip()
        value = value.strip('"').strip("'")
        if value:
            return value
        if default is not None:
            return default
        if not required:
            return ""


def prompt_existing_path(
    label: str,
    default: str | None = None,
    kinds: tuple[str, ...] = ("file",),
) -> str:
    while True:
        value = prompt_text(label, default=default, required=True)
        path = Path(value)
        if "file" in kinds and path.is_file():
            return str(path)
        if "dir" in kinds and path.is_dir():
            return str(path)
        expected = " or ".join(kinds)
        print(f"Path does not exist or is not a valid {expected}: {path}")


def prompt_choice(label: str, choices: list[str], default: str | None = None) -> str:
    display = "/".join(choices)
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"{label} ({display}){suffix}: ").strip().lower()
        if not value and default is not None:
            return default
        if value in choices:
            return value
        print(f"Please choose one of: {', '.join(choices)}")


def parse_title_indices(raw: str) -> set[int]:
    return {int(item.strip()) for item in raw.split(",") if item.strip()}


def resolve_profile_path(raw: str) -> Path:
    path = Path(str(raw).strip().strip('"').strip("'"))
    if path.is_file():
        return path
    if path.is_dir():
        candidate = path / f"{path.name}.pt"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Invalid voice profile path. Please provide a .pt file or a directory containing <dirname>.pt"
    )


def validate_existing_path(
    raw: str, label: str, kinds: tuple[str, ...] = ("file",)
) -> str:
    path = Path(str(raw).strip().strip('"').strip("'"))
    if ("file" in kinds and path.is_file()) or ("dir" in kinds and path.is_dir()):
        return str(path)
    expected = " or ".join(kinds)
    raise FileNotFoundError(f"Invalid {label} ({expected}): {path}")


def validate_existing_file(raw: str, label: str) -> str:
    path = Path(str(raw).strip().strip('"').strip("'"))
    if not path.is_file():
        raise FileNotFoundError(f"Invalid {label}: {path}")
    return str(path)


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


def needs_text_inputs(run_mode: str, target_stage: str | None) -> bool:
    return run_mode == "full" or (run_mode == "only" and target_stage == "text")


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


def is_text_file_input(path_text: str | None) -> bool:
    return bool(path_text) and Path(str(path_text)).suffix.lower() == ".txt"


def resolve_initial_args(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    run_mode = args.run_mode
    target_stage = args.target_stage

    config["ppt"] = None
    config["page"] = args.page
    config["video"] = None
    config["title_mode"] = args.title_mode
    config["title_indices"] = set()
    config["spoken_json"] = None
    config["timeline"] = None
    config["segments_manifest"] = None
    config["outro_profile"] = None
    config["input_mode"] = "document"
    config["reference_text"] = getattr(args, "reference_text", None)
    config["enable_ocr"] = bool(getattr(args, "enable_ocr", False))

    if needs_text_inputs(run_mode, target_stage):
        if args.ppt:
            config["ppt"] = validate_existing_file(args.ppt, "document path")
        elif args.video:
            config["ppt"] = None
            config["input_mode"] = "video_context"
        else:
            config["ppt"] = prompt_existing_path("Document path (.pptx or .txt)")

        if config["input_mode"] == "video_context":
            config["page"] = 1
            config["title_mode"] = "none"
            config["title_indices"] = set()
        elif is_text_file_input(config["ppt"]):
            config["page"] = args.page or 1
        else:
            config["page"] = args.page or int(prompt_text("Page number"))
            title_mode = args.title_mode or prompt_choice(
                "Title mode\n- first: treat the first paragraph as title\n- none: treat all paragraphs as narration\n- manual: choose title paragraph indices manually\nChoice",
                ["first", "none", "manual"],
                default="first",
            )
            config["title_mode"] = title_mode
            if title_mode == "manual":
                raw_indices = args.title_indices or prompt_text(
                    "Title paragraph indices (comma separated)", default="1"
                )
            elif title_mode == "first":
                raw_indices = "1"
            else:
                raw_indices = ""
            config["title_indices"] = parse_title_indices(raw_indices)
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
        if not profile:
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
    if cover_image is None and needs_cover_options(run_mode, target_stage):
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
    elif cover_image is not None and config.get("input_mode") == "video_context":
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
    if outro_image is not None and outro_audio is None and not outro_text:
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
    if outro_image is None and needs_outro_options(run_mode, target_stage):
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
    config["probe_mode"] = args.probe_mode
    if (
        run_mode == "full"
        or target_stage in {"timeline"}
        or (run_mode == "from" and target_stage in {"profile", "voice"})
    ):
        config["probe_times"] = args.probe_times or prompt_text(
            "Initial probe times (comma separated, keyframe times)",
            default="0,10,20,30",
        )
        env_key_name = (
            "GEMINI_API_KEY" if config.get("input_mode") == "video_context" else "MAAS_API_KEY"
        )
        config["api_key"] = args.api_key or read_env_key(env_key_name)
        if not config["api_key"]:
            config["api_key"] = prompt_text(env_key_name, required=True)
    else:
        config["probe_times"] = args.probe_times
        env_key_name = (
            "GEMINI_API_KEY" if config.get("input_mode") == "video_context" else "MAAS_API_KEY"
        )
        config["api_key"] = args.api_key or read_env_key(env_key_name)
    return config


def summarize_initial_inputs(
    config: dict[str, Any], run_mode: str, target_stage: str | None
) -> list[str]:
    lines = [f"run_mode: {run_mode}"]
    if target_stage is not None:
        lines.append(f"target_stage: {target_stage}")

    if needs_text_inputs(run_mode, target_stage):
        lines.append(
            f"document: {config.get('ppt') or '[video-context scripting]'}"
        )
        lines.append(f"page: {config.get('page')}")
        lines.append(f"title_mode: {config.get('title_mode')}")
        if config.get("title_mode") == "manual":
            lines.append(
                "title_indices: "
                + ",".join(str(item) for item in sorted(config.get("title_indices", [])))
            )

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

    if config.get("probe_times"):
        lines.append(f"probe_times: {config.get('probe_times')}")
    lines.append(
        "api_key: "
        + ("set" if config.get("api_key") else "not set")
    )
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
    args.ppt = config.get("ppt")
    args.page = config.get("page")
    args.video = config.get("video")
    args.title_mode = config.get("title_mode")
    title_indices = sorted(config.get("title_indices", []))
    args.title_indices = ",".join(str(item) for item in title_indices) if title_indices else None
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
    args.probe_mode = config.get("probe_mode")
    args.probe_times = config.get("probe_times")
    args.api_key = config.get("api_key")


def available_edit_sections(
    config: dict[str, Any], run_mode: str, target_stage: str | None
) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    if needs_text_inputs(run_mode, target_stage):
        sections.append(("text", "text inputs"))
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
    if config.get("probe_times") or run_mode == "full" or target_stage in {"timeline"}:
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
    if section == "text":
        args.ppt = None
        args.page = None
        args.title_mode = None
        args.title_indices = None
        args.reference_text = None
        args.enable_ocr = False
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
        args.probe_times = None
        args.api_key = None
        return


def resolve_run_plan(args: argparse.Namespace) -> tuple[str, str | None]:
    if args.only_stage and args.from_stage:
        raise ValueError("Use either --only-stage or --from-stage, not both.")

    if args.only_stage:
        target_stage = STAGE_ALIASES.get(args.only_stage, args.only_stage)
        if target_stage in {"text", "profile", "voice", "timeline", "compose"}:
            return "only", target_stage
        raise ValueError(
            "Unsupported --only-stage value. Use one of: 1/text, 2/profile, 3/voice, 4/timeline, 5/compose"
        )

    if args.from_stage:
        target_stage = STAGE_ALIASES.get(args.from_stage, args.from_stage)
        if target_stage in {"text", "profile", "voice", "timeline", "compose"}:
            return "from", target_stage
        raise ValueError(
            "Unsupported --from-stage value. Use one of: 1/text, 2/profile, 3/voice, 4/timeline, 5/compose"
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
        default="voice" if run_mode == "from" else "text",
    )
    return run_mode, target_stage


def ensure_profile_path(config: dict[str, Any]) -> Path:
    profile = config.get("profile")
    if not profile:
        raise ValueError("Voice profile path is required for this stage.")
    return Path(profile)


def ensure_spoken_json_path(
    config: dict[str, Any], stage1_result: dict[str, Any] | None = None
) -> Path:
    if stage1_result is not None:
        return Path(stage1_result["spoken_path"])
    spoken_json = config.get("spoken_json")
    if not spoken_json:
        raise ValueError("Spoken JSON path is required for this stage.")
    return Path(spoken_json)


def ensure_timeline_path(config: dict[str, Any]) -> Path:
    timeline = config.get("timeline") or config.get("timeline_output")
    if not timeline:
        raise ValueError("Timeline JSON path is required for compose stage.")
    return Path(timeline)


def ensure_segments_manifest_path(config: dict[str, Any]) -> Path:
    manifest = config.get("segments_manifest")
    if not manifest:
        raise ValueError("Segments manifest path is required for compose stage.")
    return Path(manifest)


def read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_page_from_spoken_json(spoken_json: Path) -> int:
    payload = read_json_file(spoken_json)
    return int(payload.get("page", 1))


def stage_banner(index: int, total: int, name: str) -> None:
    print()
    print(f"[{index}/{total}] {name}")


def ask_continue_or_stop(stage_name: str, allow_back: bool = True) -> str:
    if allow_back:
        return prompt_choice(
            f"{stage_name} review action\n- c: continue to the next stage\n- b: go back to the previous stage\n- s: stop here\nChoice",
            ["c", "b", "s"],
            default="c",
        )
    return prompt_choice(
        f"{stage_name} review action\n- c: continue to the next stage\n- s: stop here\nChoice",
        ["c", "s"],
        default="c",
    )


def show_stage1_summary(result: dict[str, Any]) -> None:
    print("Stage 1 completed.")
    print(f"extracted_json: {result['extracted_path']}")
    print(f"spoken_json:    {result['spoken_path']}")
    print("Review suggestions:")
    print("- Check page_XX.spoken.json")
    print("- Edit paragraphs[].spoken_text if wording needs adjustment")
    print(
        "- If title handling is wrong, rerun Stage 1 with title mode or title indices"
    )


def show_profile_summary(profile_path: Path) -> None:
    print("Voice profile created.")
    print(f"profile_path: {profile_path}")
    print("Proceeding to voice generation for actual audio review...")


def show_stage2_summary(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    segments_dir = manifest_path.parent / "segments"
    print("Stage 2 completed.")
    print(f"manifest: {manifest_path}")
    print(f"segments_dir: {segments_dir}")
    print("Available paragraphs:")
    print(
        ", ".join(str(item["paragraph_index"]) for item in payload.get("segments", []))
    )
    print("Review suggestions:")
    print("- If wording is wrong, edit page_XX.spoken.json -> paragraphs[].spoken_text")
    print("- If wording is correct but audio sounds bad, regenerate by paragraph index")
    print("- Paragraph audio files are stored under segments_dir")
    return payload


def ask_stage2_action(allow_back: bool = True) -> str:
    if allow_back:
        return prompt_choice(
            "Stage 2 review action\n- c: continue to the next stage\n- r: regenerate one or more paragraphs\n- b: go back to Stage 1\n- s: stop here\nChoice",
            ["c", "r", "b", "s"],
            default="c",
        )
    return prompt_choice(
        "Stage 2 review action\n- c: continue to the next stage\n- r: regenerate one or more paragraphs\n- s: stop here\nChoice",
        ["c", "r", "s"],
        default="c",
    )


def ask_regenerate_target() -> str:
    return prompt_text(
        "Enter paragraph indices to regenerate (comma separated or 'all')",
        required=True,
    )


def parse_paragraph_indices(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one paragraph index is required.")
    return values


def ask_voice_generation_scope(
    config: dict[str, Any],
    prompt_if_missing: bool = True,
) -> tuple[list[int] | None, float | None]:
    raw = config.get("paragraphs")
    if raw is None:
        if not prompt_if_missing:
            return None, None
        raw = prompt_text(
            "Paragraph indices to generate (comma separated, empty means all)",
            default="",
            required=False,
        ).strip()
    else:
        raw = str(raw).strip()
    if not raw:
        if config.get("volume_gain") is not None:
            print(
                f"Applying volume gain {float(config['volume_gain'])} to all generated paragraphs."
            )
        return None, None
    paragraph_indices = parse_paragraph_indices(raw)
    if config.get("volume_gain") is not None:
        print(
            "Applying volume gain "
            f"{float(config['volume_gain'])} to selected paragraphs: {','.join(str(item) for item in paragraph_indices)}"
        )
        return paragraph_indices, float(config["volume_gain"])
    return paragraph_indices, ask_regenerate_volume_gain()


def ask_regenerate_volume_gain() -> float | None:
    raw = prompt_text(
        "Optional volume gain for this regeneration (e.g. 0.9, 1.1, default empty)",
        default="",
        required=False,
    ).strip()
    if not raw:
        return None
    gain = float(raw)
    if gain <= 0:
        raise ValueError("Volume gain must be greater than 0.")
    return gain


def show_stage3_summary(timeline_path: Path) -> dict[str, Any]:
    payload = json.loads(timeline_path.read_text(encoding="utf-8"))
    print("Stage 3 completed.")
    print(f"timeline: {timeline_path}")
    print(f"status: {payload.get('status')}")
    print(f"missing: {payload.get('missing_paragraph_indices', [])}")
    print("Review suggestions:")
    print("- Edit page_XX.timeline.final.json -> segments[].start")
    print("- For missing paragraphs, set matched=true and provide start")
    print("- end_hint is optional and only used as a reference")
    return payload


def show_stage4_summary(output_video: Path, output_dir: Path) -> None:
    print("Stage 4 completed.")
    print(f"final_video: {output_video}")
    print(f"output_dir:   {output_dir}")
    print("Review suggestions:")
    print("- If audio quality is wrong, go back to Stage 2")
    print("- If timing is wrong, go back to Stage 3 and adjust starts")


def default_stage1_output_dir(config: dict[str, Any]) -> Path | None:
    if config.get("stage1_output_dir"):
        return Path(config["stage1_output_dir"])
    return None


def default_timeline_output(spoken_json: Path, config: dict[str, Any]) -> Path:
    if config.get("timeline_output"):
        return Path(config["timeline_output"])
    page = int(config.get("page") or infer_page_from_spoken_json(spoken_json))
    return spoken_json.parent / f"page_{page:02d}.timeline.final.json"


def default_timeline_debug_dir(
    config: dict[str, Any], spoken_json: Path
) -> Path | None:
    if config.get("timeline_debug_dir"):
        return Path(config["timeline_debug_dir"])
    page = int(config.get("page") or infer_page_from_spoken_json(spoken_json))
    return OUTPUTS_DIR / "timeline_debug" / f"page_{page:02d}_debug"


def default_compose_output_dir(source_path: Path, config: dict[str, Any]) -> Path:
    if config.get("compose_output_dir"):
        return Path(config["compose_output_dir"])
    page_token = source_path.stem.replace(".timeline", "")
    page_name = f"{slugify(page_token, max_len=36)}_{slugify(Path(config['video']).stem, max_len=36)}"
    return OUTPUTS_DIR / "composed" / page_name


def run_stage1(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("input_mode") == "video_context":
        output_dir = default_stage1_output_dir(config) or (
            OUTPUTS_DIR / "scripts" / slugify(Path(config["video"]).stem, max_len=60)
        )
        debug_dir = (
            Path(config["timeline_debug_dir"])
            if config.get("timeline_debug_dir")
            else OUTPUTS_DIR
            / "timeline_debug"
            / slugify(Path(config["video"]).stem, max_len=36)
        )
        return run_video_script_generate(
            video=Path(config["video"]),
            output_dir=output_dir,
            debug_dir=debug_dir,
            gemini_api_key=config.get("api_key"),
            reference_text_path=Path(config["reference_text"]) if config.get("reference_text") else None,
            cover_image=Path(config["cover_image"]) if config.get("cover_image") else None,
            cover_duration_sec=config.get("cover_duration_sec"),
            enable_ocr=bool(config.get("enable_ocr", False)),
        )
    return prepare_ppt_page(
        ppt_path=Path(config["ppt"]),
        page=int(config["page"]),
        title_indices=config["title_indices"],
        output_dir=default_stage1_output_dir(config),
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
            else None,
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
            else None,
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


def run_stage3(config: dict[str, Any], spoken_json: Path) -> Path:
    output = default_timeline_output(spoken_json, config)
    run_timeline_align(
        video=Path(config["video"]),
        spoken_json=spoken_json,
        output=output,
        debug_dir=default_timeline_debug_dir(config, spoken_json),
        api_key=config["api_key"],
        probe_mode=config["probe_mode"],
        probe_times=config["probe_times"],
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive NarrateFlow pipeline")
    parser.add_argument("--ppt")
    parser.add_argument("--input", dest="ppt")
    parser.add_argument("--page", type=int)
    parser.add_argument("--video")
    parser.add_argument("--profile")
    parser.add_argument("--voice-name")
    parser.add_argument("--ref-audio")
    parser.add_argument("--ref-text")
    parser.add_argument("--reference-text")
    parser.add_argument("--title-mode", choices=["first", "none", "manual"])
    parser.add_argument("--title-indices")
    parser.add_argument("--stage1-output-dir")
    parser.add_argument("--profile-output-dir")
    parser.add_argument("--voice-output-dir")
    parser.add_argument("--spoken-json")
    parser.add_argument("--timeline")
    parser.add_argument("--segments-manifest")
    parser.add_argument("--timeline-output")
    parser.add_argument("--timeline-debug-dir")
    parser.add_argument("--compose-output-dir")
    parser.add_argument("--cover-image")
    parser.add_argument("--cover-duration-sec", type=float)
    parser.add_argument("--cover-paragraph-index", type=int, default=2)
    parser.add_argument("--outro-image")
    parser.add_argument("--outro-audio")
    parser.add_argument("--outro-text")
    parser.add_argument("--outro-profile")
    parser.add_argument("--paragraphs")
    parser.add_argument("--volume-gain", type=float)
    parser.add_argument(
        "--probe-mode", choices=["keyframes", "times"], default="keyframes"
    )
    parser.add_argument("--probe-times")
    parser.add_argument("--api-key")
    parser.add_argument("--enable-ocr", action="store_true")
    parser.add_argument(
        "--only-stage",
        help="Run only one stage. Supported values: 1/text, 2/profile, 3/voice, 4/timeline, 5/compose",
    )
    parser.add_argument(
        "--from-stage",
        help="Start from one stage and continue. Supported values: 1/text, 2/profile, 3/voice, 4/timeline, 5/compose",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_mode, target_stage = resolve_run_plan(args)
    args.run_mode = run_mode
    args.target_stage = target_stage
    while True:
        config = resolve_initial_args(args)
        sync_config_to_args(args, config)
        input_action = confirm_initial_inputs(config, run_mode, target_stage)
        if input_action == "c":
            break
        if input_action == "s":
            return
        section = ask_edit_section(config, run_mode, target_stage)
        clear_args_for_section(args, section)

    total = 5

    if run_mode == "only":
        if target_stage == "text":
            stage_banner(1, total, "Text Processing")
            stage1_result = run_stage1(config)
            show_stage1_summary(stage1_result)
            return
        if target_stage == "profile":
            stage_banner(2, total, "Voice Profile Generation")
            profile_path = run_stage2_profile(config)
            config["profile"] = str(profile_path)
            show_profile_summary(profile_path)
            return
        if target_stage == "voice":
            stage_banner(3, total, "Voice Generation")
            paragraph_indices, volume_gain = ask_voice_generation_scope(config)
            manifest_path = run_stage2_voice(
                config,
                ensure_spoken_json_path(config),
                ensure_profile_path(config),
                paragraph_indices=paragraph_indices,
                volume_gain=volume_gain,
            )
            while True:
                show_stage2_summary(manifest_path)
                action = prompt_choice(
                    "Stage 2 review action\n- r: regenerate one or more paragraphs\n- s: stop here\nChoice",
                    ["r", "s"],
                    default="s",
                )
                if action == "s":
                    return
                target = ask_regenerate_target()
                volume_gain = ask_regenerate_volume_gain()
                manifest_path = rerun_stage2_for_target(
                    config,
                    ensure_spoken_json_path(config),
                    ensure_profile_path(config),
                    target,
                    volume_gain=volume_gain,
                )
        if target_stage == "timeline":
            stage_banner(4, total, "Timeline Alignment")
            timeline_path = run_stage3(config, ensure_spoken_json_path(config))
            show_stage3_summary(timeline_path)
            return
        if target_stage == "compose":
            stage_banner(5, total, "Video Composition")
            composed_path = run_stage4(
                config,
                ensure_timeline_path(config),
                ensure_segments_manifest_path(config),
            )
            show_stage4_summary(composed_path, composed_path.parent)
            return

    stage1_result: dict[str, Any] | None = None
    manifest_path: Path | None = None
    timeline_path: Path | None = None

    if run_mode == "full":
        while True:
            stage_banner(1, total, "Text Processing")
            stage1_result = run_stage1(config)
            show_stage1_summary(stage1_result)
            action = ask_continue_or_stop("Stage 1")
            if action == "s":
                return
            if action == "c":
                break

        while True:
            profile_path = Path(config["profile"]) if config.get("profile") else None
            if profile_path is None:
                stage_banner(2, total, "Voice Profile Generation")
                profile_path = run_stage2_profile(config)
                config["profile"] = str(profile_path)
                show_profile_summary(profile_path)
            else:
                print()
                print(
                    f"[2/{total}] Voice Profile Generation (skipped, using existing profile)"
                )
                print(f"profile_path: {profile_path}")

            stage_banner(3, total, "Voice Generation")
            paragraph_indices, volume_gain = ask_voice_generation_scope(
                config,
                prompt_if_missing=False,
            )
            manifest_path = run_stage2_voice(
                config,
                ensure_spoken_json_path(config, stage1_result),
                profile_path,
                paragraph_indices=paragraph_indices,
                volume_gain=volume_gain,
            )
            while True:
                show_stage2_summary(manifest_path)
                action = ask_stage2_action(allow_back=True)
                if action == "c":
                    break
                if action == "s":
                    return
                if action == "b":
                    break
                target = ask_regenerate_target()
                volume_gain = ask_regenerate_volume_gain()
                manifest_path = rerun_stage2_for_target(
                    config,
                    ensure_spoken_json_path(config, stage1_result),
                    profile_path,
                    target,
                    volume_gain=volume_gain,
                )
            if action == "b":
                continue
            break

    if run_mode == "from" and target_stage in {"voice", "timeline"}:
        stage1_result = {"spoken_path": ensure_spoken_json_path(config)}

    if run_mode == "from" and target_stage in {"profile", "voice"}:
        while True:
            profile_path = Path(config["profile"]) if config.get("profile") else None
            if profile_path is None:
                stage_banner(2, total, "Voice Profile Generation")
                profile_path = run_stage2_profile(config)
                config["profile"] = str(profile_path)
                show_profile_summary(profile_path)
            else:
                print()
                print(
                    f"[2/{total}] Voice Profile Generation (skipped, using existing profile)"
                )
                print(f"profile_path: {profile_path}")

            stage_banner(3, total, "Voice Generation")
            paragraph_indices, volume_gain = ask_voice_generation_scope(config)
            manifest_path = run_stage2_voice(
                config,
                ensure_spoken_json_path(config, stage1_result),
                profile_path,
                paragraph_indices=paragraph_indices,
                volume_gain=volume_gain,
            )
            while True:
                show_stage2_summary(manifest_path)
                action = ask_stage2_action(allow_back=run_mode == "full")
                if action == "c":
                    break
                if action == "s":
                    return
                if action == "b":
                    break
                target = ask_regenerate_target()
                volume_gain = ask_regenerate_volume_gain()
                manifest_path = rerun_stage2_for_target(
                    config,
                    ensure_spoken_json_path(config, stage1_result),
                    profile_path,
                    target,
                    volume_gain=volume_gain,
                )
            if action == "b":
                continue
            break

    if run_mode == "full" or (
        run_mode == "from" and target_stage in {"profile", "voice", "timeline"}
    ):
        while True:
            stage_banner(4, total, "Timeline Alignment")
            timeline_path = run_stage3(
                config, ensure_spoken_json_path(config, stage1_result)
            )
            show_stage3_summary(timeline_path)
            action = ask_continue_or_stop(
                "Stage 4",
                allow_back=not (run_mode == "from" and target_stage == "timeline"),
            )
            if action == "s":
                return
            if action == "b":
                profile_path = (
                    Path(config["profile"]) if config.get("profile") else None
                )
                stage_banner(3, total, "Voice Generation")
                paragraph_indices, volume_gain = ask_voice_generation_scope(
                    config,
                    prompt_if_missing=run_mode != "full",
                )
                manifest_path = run_stage2_voice(
                    config,
                    ensure_spoken_json_path(config, stage1_result),
                    profile_path,
                    paragraph_indices=paragraph_indices,
                    volume_gain=volume_gain,
                )
                while True:
                    show_stage2_summary(manifest_path)
                    action2 = ask_stage2_action(allow_back=False)
                    if action2 == "c":
                        break
                    if action2 == "s":
                        return
                    if action2 == "b":
                        break
                    target = ask_regenerate_target()
                    volume_gain = ask_regenerate_volume_gain()
                    manifest_path = rerun_stage2_for_target(
                        config,
                        ensure_spoken_json_path(config, stage1_result),
                        profile_path,
                        target,
                        volume_gain=volume_gain,
                    )
                if action2 == "b":
                    continue
                continue
            break

    if run_mode == "from" and target_stage == "compose":
        timeline_path = ensure_timeline_path(config)
        manifest_path = ensure_segments_manifest_path(config)

    if manifest_path is None and run_mode == "from" and target_stage == "timeline":
        manifest_path = ensure_segments_manifest_path(config)

    if manifest_path is None:
        raise ValueError("Segments manifest is required before video composition.")
    if timeline_path is None:
        raise ValueError("Timeline path is required before video composition.")

    stage_banner(5, total, "Video Composition")
    composed_path = run_stage4(config, timeline_path, manifest_path)
    show_stage4_summary(composed_path, composed_path.parent)


if __name__ == "__main__":
    main()
