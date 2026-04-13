from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voice_process.common import ROOT, load_model, save_prompt_file
from voice_clone_tool import slugify


PROFILES_DIR = ROOT / "outputs" / "voice_profiles"


def run_voice_profile(
    voice_name: str,
    ref_audio: str,
    ref_text: str | None = None,
    xvector_only: bool = False,
    device: str | None = None,
    dtype: str | None = None,
) -> Path:
    tts = load_model(device=device, dtype=dtype)
    prompt_items = tts.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=xvector_only,
    )
    profile_name = slugify(voice_name)
    profile_path = PROFILES_DIR / profile_name / f"{profile_name}.pt"
    save_prompt_file(prompt_items, profile_path)
    return profile_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 voice profile generation")
    parser.add_argument("--voice-name", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--ref-text", default=None)
    parser.add_argument("--xvector-only", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    args = parser.parse_args()

    profile_path = run_voice_profile(
        voice_name=args.voice_name,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        xvector_only=args.xvector_only,
        device=args.device,
        dtype=args.dtype,
    )
    print(profile_path)


if __name__ == "__main__":
    main()
