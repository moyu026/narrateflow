from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
    print("VLM script generation completed.")
    print(f"spoken_json:    {result['spoken_path']}")
    print("Review suggestions:")
    print("- Check page_XX.spoken.json")
    print("- Edit paragraphs[].spoken_text if wording needs adjustment")
    print("- If visual context is wrong, rerun keyframe extraction with settings adjusted")


def show_keyframe_summary(result: dict[str, Any]) -> None:
    windows = result.get("windows", [])
    keyframes = result.get("keyframes", {})
    candidates = keyframes.get("candidates", [])
    print("Keyframe extraction completed.")
    print(f"keyframes_json:      {result['keyframes_path']}")
    print(f"window_manifest:     {result['window_manifest_path']}")
    print(f"keyframe_count:      {len(candidates)}")
    print(f"window_count:        {len(windows)}")
    print("Review suggestions:")
    print("- Check timeline_debug/keyframes/")
    print("- Check window_manifest.json if boundaries look wrong")


def show_profile_summary(profile_path: Path) -> None:
    print("Voice profile created.")
    print(f"profile_path: {profile_path}")
    print("Proceeding to voice generation for actual audio review...")


def show_stage2_summary(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    segments_dir = manifest_path.parent / "segments"
    print("Voice generation completed.")
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
            "Voice generation review action\n- c: continue to the next stage\n- r: regenerate one or more paragraphs\n- b: go back to VLM script generation\n- s: stop here\nChoice",
            ["c", "r", "b", "s"],
            default="c",
        )
    return prompt_choice(
        "Voice generation review action\n- c: continue to the next stage\n- r: regenerate one or more paragraphs\n- s: stop here\nChoice",
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
    print("Timeline alignment completed.")
    print(f"timeline: {timeline_path}")
    print(f"status: {payload.get('status')}")
    print(f"missing: {payload.get('missing_paragraph_indices', [])}")
    print("Review suggestions:")
    print("- Edit page_XX.timeline.final.json -> segments[].start")
    print("- For missing paragraphs, set matched=true and provide start")
    print("- end_hint is optional and only used as a reference")
    return payload


def show_stage4_summary(output_video: Path, output_dir: Path) -> None:
    print("Video composition completed.")
    print(f"final_video: {output_video}")
    print(f"output_dir:   {output_dir}")
    print("Review suggestions:")
    print("- If audio quality is wrong, go back to Stage 2")
    print("- If timing is wrong, go back to Stage 3 and adjust starts")
