from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"
STAGE_ALIASES = {
    "1": "script",
    "2": "profile",
    "3": "voice",
    "4": "timeline",
    "5": "compose",
    "script": "script",
    "text": "script",
    "profile": "profile",
    "voice": "voice",
    "timeline": "timeline",
    "compose": "compose",
}

STAGE_SELECTION_CHOICES = ["script", "profile", "voice", "timeline", "compose"]
RUN_MODE_CHOICES = ["full", "only", "from"]
DEFAULT_VIDEO_CONFIG = ROOT / "config" / "video_mode.toml"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def slugify(value: str, max_len: int = 60) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", str(value).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return (cleaned or "untitled")[:max_len]


def video_output_root(video: str | Path) -> Path:
    return OUTPUTS_DIR / slugify(Path(video).stem, max_len=60)
