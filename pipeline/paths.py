from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
