from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pipeline.paths import (
    ensure_profile_path,
    ensure_segments_manifest_path,
    ensure_spoken_json_path,
    ensure_timeline_path,
)
from pipeline.runtime import (
    apply_pipeline_mode_config,
    ask_edit_section,
    clear_args_for_section,
    confirm_initial_inputs,
    reload_runtime_config,
    resolve_run_plan,
)
from pipeline.shared import DEFAULT_VIDEO_CONFIG
from pipeline.stages import (
    ask_only_voice_action,
    run_profile_and_voice_stages,
    run_stage1,
    run_stage1_keyframes,
    run_stage1_vlm,
    run_stage2_profile,
    run_stage3,
    run_stage4,
    run_voice_stage_with_review,
)
from pipeline.ui import (
    ask_continue_or_stop,
    ask_stage2_action,
    show_profile_summary,
    show_keyframe_summary,
    show_stage1_summary,
    show_stage3_summary,
    show_stage4_summary,
    stage_banner,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive NarrateFlow pipeline")
    parser.add_argument("--video-config", default=str(DEFAULT_VIDEO_CONFIG))
    parser.add_argument("--page", type=int)
    parser.add_argument("--video")
    parser.add_argument("--profile")
    parser.add_argument("--voice-name")
    parser.add_argument("--ref-audio")
    parser.add_argument("--ref-text")
    parser.add_argument("--reference-text")
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
    parser.add_argument("--frame-stride", type=int)
    parser.add_argument("--min-gap-sec", type=float, default=2.0)
    parser.add_argument("--global-threshold", type=float, default=12.0)
    parser.add_argument("--subtitle-threshold", type=float, default=8.0)
    parser.add_argument("--detection-max-width", type=int, default=960)
    parser.add_argument("--fill-gap-sec", type=float, default=6.0)
    parser.add_argument("--api-key")
    parser.add_argument("--enable-ocr", action="store_true")
    parser.add_argument(
        "--only-stage",
        help="Run only one stage. Supported values: script, profile, voice, timeline, compose",
    )
    parser.add_argument(
        "--from-stage",
        help="Start from one stage and continue. Supported values: script, profile, voice, timeline, compose",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    apply_pipeline_mode_config(args)
    run_mode, target_stage = resolve_run_plan(args)
    args.run_mode = run_mode
    args.target_stage = target_stage
    config: dict[str, Any] | None = None
    while True:
        config = reload_runtime_config(args, config)
        if getattr(args, "skip_initial_input_review", False):
            break
        input_action = confirm_initial_inputs(config, run_mode, target_stage)
        if input_action == "c":
            break
        if input_action == "s":
            return
        section = ask_edit_section(config, run_mode, target_stage)
        clear_args_for_section(args, section)

    total = 6

    if run_mode == "only":
        if target_stage == "script":
            config = reload_runtime_config(args, config)
            stage_banner(1, total, "Keyframe Extraction")
            prepared = run_stage1_keyframes(config)
            show_keyframe_summary(prepared)
            stage_banner(2, total, "VLM Script Generation")
            stage1_result = run_stage1_vlm(config, prepared)
            show_stage1_summary(stage1_result)
            return
        if target_stage == "profile":
            config = reload_runtime_config(args, config)
            stage_banner(3, total, "Voice Profile Generation")
            profile_path = run_stage2_profile(config)
            config["profile"] = str(profile_path)
            show_profile_summary(profile_path)
            return
        if target_stage == "voice":
            stage_banner(4, total, "Voice Generation")
            config, manifest_path, action = run_voice_stage_with_review(
                args=args,
                config=config,
                spoken_json=ensure_spoken_json_path(config),
                profile_path=ensure_profile_path(config),
                allow_back=False,
                prompt_if_missing=True,
                action_selector=ask_only_voice_action,
            )
            if action == "s":
                return
            if action == "c":
                return
            raise ValueError(f"Unexpected voice stage action: {action}")
        if target_stage == "timeline":
            config = reload_runtime_config(args, config)
            stage_banner(5, total, "Timeline Alignment")
            timeline_path = run_stage3(config, ensure_spoken_json_path(config))
            show_stage3_summary(timeline_path)
            return
        if target_stage == "compose":
            config = reload_runtime_config(args, config)
            stage_banner(6, total, "Video Composition")
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
            config = reload_runtime_config(args, config)
            stage_banner(1, total, "Keyframe Extraction")
            prepared = run_stage1_keyframes(config)
            show_keyframe_summary(prepared)
            action = ask_continue_or_stop("Stage 1", allow_back=False)
            if action == "s":
                return

            config = reload_runtime_config(args, config)
            stage_banner(2, total, "VLM Script Generation")
            stage1_result = run_stage1_vlm(config, prepared)
            show_stage1_summary(stage1_result)
            action = ask_continue_or_stop("Stage 2")
            if action == "s":
                return
            if action == "c":
                break
            if action == "b":
                continue

        while True:
            config, manifest_path, action = run_profile_and_voice_stages(
                args=args,
                config=config,
                total=total,
                spoken_json=ensure_spoken_json_path(config, stage1_result),
                allow_back=True,
                prompt_if_missing=False,
            )
            if action == "b":
                continue
            if action == "s":
                return
            break

    if run_mode == "from" and target_stage in {"voice", "timeline"}:
        stage1_result = {"spoken_path": ensure_spoken_json_path(config)}

    if run_mode == "from" and target_stage in {"profile", "voice"}:
        while True:
            config, manifest_path, action = run_profile_and_voice_stages(
                args=args,
                config=config,
                total=total,
                spoken_json=ensure_spoken_json_path(config, stage1_result),
                allow_back=run_mode == "full",
                prompt_if_missing=True,
            )
            if action == "b":
                continue
            if action == "s":
                return
            break

    if run_mode == "full" or (
        run_mode == "from" and target_stage in {"profile", "voice", "timeline"}
    ):
        while True:
            config = reload_runtime_config(args, config)
            stage_banner(5, total, "Timeline Alignment")
            timeline_path = run_stage3(
                config, ensure_spoken_json_path(config, stage1_result)
            )
            show_stage3_summary(timeline_path)
            action = ask_continue_or_stop(
                "Stage 5",
                allow_back=not (run_mode == "from" and target_stage == "timeline"),
            )
            if action == "s":
                return
            if action == "b":
                config = reload_runtime_config(args, config)
                profile_path = (
                    Path(config["profile"]) if config.get("profile") else None
                )
                stage_banner(4, total, "Voice Generation")
                config, manifest_path, action2 = run_voice_stage_with_review(
                    args=args,
                    config=config,
                    spoken_json=ensure_spoken_json_path(config, stage1_result),
                    profile_path=profile_path,
                    allow_back=False,
                    prompt_if_missing=run_mode != "full",
                    action_selector=ask_stage2_action,
                )
                if action2 == "b":
                    continue
                if action2 == "s":
                    return
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

    stage_banner(6, total, "Video Composition")
    config = reload_runtime_config(args, config)
    composed_path = run_stage4(config, timeline_path, manifest_path)
    show_stage4_summary(composed_path, composed_path.parent)


if __name__ == "__main__":
    main()
