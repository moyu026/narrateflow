# NarrateFlow

NarrateFlow is a human-in-the-loop pipeline for turning a source video into a narrated, dubbed video.

It samples keyframes from the video, asks a vision-language model to draft concise narration windows, generates paragraph-level voice audio, aligns the narration to the video timeline, and renders the final composed video.

## Environment

| Component | Requirement | Notes |
|---|---|---|
| Python | 3.13 | Recommended runtime |
| FFmpeg / FFprobe | Required | Used for frame extraction and video composition |
| CUDA | Recommended | Speeds up local TTS inference |
| Local TTS backend | Qwen-TTS | Used in `voice_process` |
| VL backend | Gemini via Google GenAI API | Used in `timeline_align` |
| API key | `GEMINI_API_KEY` | Required for timeline alignment |

## Installation

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Notes:

- `requirements.txt` covers the Python package dependencies used by the current pipeline.
- For GPU acceleration, make sure your `torch` / `torchaudio` / `torchvision` installation matches your CUDA environment.
- `ffmpeg` / `ffprobe` must be installed separately and available in `PATH`.
- `sox` is optional but recommended. If unavailable, speed adjustment falls back to `librosa`.


## Recommended Usage

The recommended way to run NarrateFlow is through the pipeline entrypoint:

```bash
python run_pipeline.py
```

The default configuration file is `config/video_mode.toml`. Fill in the video path, Gemini key or `GEMINI_API_KEY`, and voice profile settings before running.

The pipeline supports three execution modes:

1. `full`: run the complete pipeline from Stage 1 to Stage 6
2. `only`: run only one stage
3. `from`: start from a stage and continue forward

Stage names:

1. Keyframe extraction: sample keyframes and build `window_manifest.json`
2. VLM script generation: call Gemini on the prepared windows and write narration JSON
3. Voice profile generation
4. Voice generation
5. Timeline JSON generation
6. Final video composition

Default outputs are grouped by video:

```text
outputs/
└── <video_stem>/
    ├── scripts/
    │   └── page_01.spoken.json
    ├── timeline_debug/
    │   ├── keyframes/
    │   ├── keyframes.json
    │   ├── window_manifest.json
    │   ├── window_script_drafts.json
    │   ├── gemini_batch_requests.json
    │   └── gemini_batch_responses.json
    ├── timeline/
    │   ├── page_01.timeline.final.json
    │   └── debug/
    ├── voice/
    │   ├── segments_manifest.json
    │   └── segments/
    └── composed/
        ├── page_audio.wav
        ├── page_composed.mp4
        └── page_plan.json
```

If a cover image should be shown before the main video starts, the interactive runner can now ask for:

- whether to enable a cover intro
- `cover image path`
- `cover paragraph index`
- optional cover duration override

The interactive runner can also ask for an outro page:

- whether to enable an outro page
- `outro image path`
- whether a fixed slogan audio already exists
- otherwise, a fixed slogan text that can be synthesized with the current voice profile

## Example Interactive Flow

Below is a simplified example of a full video-mode run.

### Input Collection

```text
Run mode
- full: run the full pipeline
- only: run only one stage
- from: start from one stage and continue
Choice (full/only/from) [full]: full

Target video path: <path/to/example.mp4>

Do you already have a voice profile file (y/n) [y]: y
Voice profile path (.pt file or profile directory): outputs/voice_profiles/reference_voice

Do you want to prepend a cover image before the main video (y/n) [n]: y
Cover image path: <path/to/cover.png>
Cover paragraph index [1]: 1
Optional cover duration in seconds (empty means use cover paragraph audio duration):

Do you want to append an outro page after the main video (y/n) [n]: y
Outro image path: <path/to/outro.png>
Do you already have a fixed outro slogan audio (y/n) [y]: n
Outro slogan text: <your fixed slogan text>
```

You can also run a single stage from the config:

```bash
python run_pipeline.py --only-stage script
```

### Stage 1. Keyframe Extraction

**Output and Review**

Check:
- sampled keyframes
- generated narration windows

Edit if needed:
- keyframe settings in `config/video_mode.toml`

```text
[1/6] Keyframe Extraction
Keyframe extraction completed.
keyframes_json:      outputs/<video_stem>/timeline_debug/keyframes.json
window_manifest:     outputs/<video_stem>/timeline_debug/window_manifest.json
keyframe_count:      32
window_count:        8

Stage 1 review action
- c: continue to the next stage
- s: stop here
Choice (c/s) [c]: c
```

### Stage 2. VLM Script Generation

**Output and Review**

Check:
- spoken narration wording
- whether large on-screen annotation text was interpreted correctly
- whether each narration paragraph starts at the expected window

Edit if needed:
- `page_01.spoken.json -> paragraphs[].spoken_text`

```text
[2/6] VLM Script Generation
VLM script generation completed.
spoken_json:    outputs/<video_stem>/scripts/page_01.spoken.json

Stage 2 review action
- c: continue to the next stage
- b: go back to the previous stage
- s: stop here
Choice (c/b/s) [c]: c
```

### Stage 3. Voice Profile Generation

**Output**

```text
[3/6] Voice Profile Generation (skipped, using existing profile)
profile_path: outputs/voice_profiles/reference_voice/reference_voice.pt
```

### Stage 4. Voice Generation

**Output and Review**

Check:
- paragraph-level audio quality
- omitted or weakly spoken words
- sentence endings

The pipeline now supports generating all narration paragraphs or only selected paragraphs.

For selected paragraphs, you can also apply an optional volume gain.

Examples:
- empty input: generate all paragraphs
- `3`: generate paragraph 3 only
- `3,5,7`: generate selected paragraphs

Equivalent CLI options:
- `--paragraphs 3,5,7`
- `--volume-gain 1.1`

Edit or regenerate if needed:
- edit `page_XX.spoken.json -> paragraphs[].spoken_text` if wording is wrong
- regenerate by paragraph index if wording is correct but audio sounds bad

```text
[4/6] Voice Generation
Paragraph indices to generate (comma separated, empty means all): 4,7
Optional volume gain for this regeneration (e.g. 0.9, 1.1, default empty): 1.1

Voice generation completed.
manifest: outputs/<video_stem>/voice/segments_manifest.json
segments_dir: outputs/<video_stem>/voice/segments
Available paragraphs:
2, 3, 4, 5, 6, 7

Voice generation review action
- c: continue to the next stage
- r: regenerate one or more paragraphs
- s: stop here
Choice (c/r/s) [c]: r
Enter paragraph indices to regenerate (comma separated or 'all'): 4,7
```

### Stage 5. Timeline Alignment

**Output and Review**

Check:
- paragraph starts
- missing paragraphs
- ordering issues

If a cover intro is enabled, the cover paragraph does not need a body timeline start.
For example, when `cover_paragraph_index=2`, paragraph 2 is treated as the intro segment and the body timeline starts from paragraph 3.

Edit if needed:
- `page_XX.timeline.final.json -> segments[].start`
- for missing paragraphs, set `matched=true` and provide `start`

```text
[5/6] Timeline Alignment
Timeline alignment completed.
timeline: outputs/<video_stem>/timeline/page_01.timeline.final.json
status: complete
missing: []

Stage 5 review action
- c: continue to the next stage
- s: stop here
Choice (c/s) [c]: c
```

### Stage 6. Video Composition

**Output and Review**

Check:
- final pacing
- retiming quality
- audio-video alignment

If a cover intro is enabled, Stage 6 will:
- prepend the cover image as a static intro clip
- use the selected cover paragraph audio at the beginning
- shift the main video body after the intro duration

If an outro page is enabled, Stage 6 will:
- append a static outro page after the main video
- use a fixed slogan audio if provided
- otherwise generate the slogan audio from the current voice profile and append it at the end

If something is wrong:
- go back to Stage 4 for audio issues
- go back to Stage 5 for timing issues

```text
[6/6] Video Composition
Video composition completed.
final_video: outputs/<video_stem>/composed/page_composed.mp4
output_dir:   outputs/<video_stem>/composed
```

## Current Behavior

- video-context script generation is split into keyframe extraction and VLM script generation stages
- video windows use left-closed, right-open frame assignment, so boundary keyframes belong to the next window
- narration character budgets scale with keyframe count: `50 * keyframe_count` per window
- narration prompts ask the VLM to prioritize large on-screen text because it is often annotation text
- paragraph-level audio generation
- optional paragraph selection and volume gain during Stage 4 voice generation
- profile path accepts either a `.pt` file or a profile directory containing `<dirname>.pt`
- optional cover intro support with `cover_image` and `cover_paragraph_index`
- when a cover intro is enabled, the cover paragraph is excluded from body timeline alignment and the body starts from the next paragraph
- optional outro page support with either fixed slogan audio or generated slogan audio
- start-driven timeline semantics
- local video retiming instead of truncating audio
- current composition defaults:
  - `buffer_sec = 1.2`
  - `tail_buffer_sec = 1.5`
  - `audio_tail_pad_sec = 0.5`

## Limitations

- timeline alignment quality depends on UI visibility, subtitle availability, and visual distinction between adjacent segments
- human review is still recommended for production-quality output
- some sentence endings may require spoken-text rewriting for better TTS delivery
- generated narration can still require manual adjustment when UI text appears late within a merged window

## Project Structure

- `outputs/voice_profiles/`: saved voice profiles `.pt`
- `outputs/<video_stem>/scripts/`: generated `spoken.json`
- `outputs/<video_stem>/timeline_debug/`: Stage 1 keyframes/window manifests and Stage 2 prompts/model responses
- `outputs/<video_stem>/timeline/`: final timeline JSON and Stage 5 debug files
- `outputs/<video_stem>/voice/`: generated voice segment WAVs and manifest
- `outputs/<video_stem>/composed/`: final composed video and composition artifacts
- `pipeline/`: reusable page-level workflow scripts and older tooling
- `sample/`: reference examples or experiments

## Roadmap

- continue refining timeline alignment quality and probe strategy
- keep simplifying `run_pipeline.py` interaction without making the workflow more rigid
- improve README examples with more realistic sample commands and outputs
- add better review and recovery tools for paragraph-level regeneration and timeline fixing
