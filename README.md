# NarrateFlow

NarrateFlow is a human-in-the-loop pipeline for turning PPTs or documents into dubbed videos.

It extracts narration text, generates paragraph-level voice audio, builds the video timeline from script timestamps, and renders the final dubbed video.

## Environment

| Component | Requirement | Notes |
|---|---|---|
| Python | 3.13 | Recommended runtime |
| FFmpeg / FFprobe | Required | Used for video composition and duration probing |
| CUDA | Recommended | Speeds up local TTS inference |
| Local TTS backend | Qwen-TTS | Used in `voice_process` |

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

The recommended way to run NarrateFlow is through the config-driven pipeline entrypoint:

```bash
python run_pipeline.py
```

By default, `run_pipeline.py` reads `pipeline_config.toml`. You can use a different file with:

```bash
python run_pipeline.py --config path/to/pipeline_config.toml
```

The config supports three execution modes:

1. `all`: run the complete pipeline from Stage 1 to Stage 5
2. `only`: run only one stage
3. `from`: start from a stage and continue forward

Keep exactly one mode section active in `pipeline_config.toml`. The default template leaves `[all]` active and comments out `[only]` and `[from]`. Edit the active section values before running.

Stage 1 now supports both `.pptx` and `.txt` inputs. For plain text input:

- use a `.txt` file as the document source
- the pipeline treats it as a single-page script source
- if the file contains blank lines, blank lines are used as paragraph boundaries
- otherwise, each non-empty line is treated as one paragraph
- each narration paragraph must start with a timestamp such as `0.0s,正文` or `1.5s，正文`
- timestamp prefixes are removed before TTS and saved as `timeline_start_sec`

If a cover image should be shown before the main video starts, set these fields:

- `cover image path`
- `cover paragraph index`
- optional cover duration override

The config can also enable an outro page:

- `outro image path`
- either a fixed slogan audio path
- or fixed slogan text that can be synthesized with the current voice profile

## Example Config Flow

Below is a simplified `[all]` config example.

### Input Collection

```toml
[all]
ppt = "path/to/example.pptx"
page = 5
video = "path/to/example.mp4"
title_mode = "first"
title_indices = "1"
profile = "outputs/voice_profiles/reference_voice"
```

For a plain text script, set `ppt` to the `.txt` path and leave `page = 1`.

Example timestamped text:

```text
0.0s,这是第一段内容

1.5s,这是第二段内容
```

To run only Stage 1, comment out `[all]`, uncomment `[only]`, and set `stage = "text"`.

### Stage 1. Text Processing

**Output and Review**

Check:
- paragraph extraction
- title handling
- spoken narration wording
- whether header/footer-like short text has been filtered as expected

Edit if needed:
- `page_XX.spoken.json -> paragraphs[].spoken_text`

```text
[1/5] Text Processing
Stage 1 completed.
extracted_json: outputs/scripts/<page_name>/page_05.extracted.json
spoken_json:    outputs/scripts/<page_name>/page_05.spoken.json

Stage 1 review action
- c: continue to the next stage
- s: stop here
Choice (c/s) [c]: c
```

### Stage 2. Voice Profile Generation

**Output**

```text
[2/5] Voice Profile Generation (skipped, using existing profile)
profile_path: outputs/voice_profiles/reference_voice/reference_voice.pt
```

### Stage 3. Voice Generation

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

Equivalent config fields:
- `paragraphs = "3,5,7"`
- `volume_gain = 1.1`

Edit or regenerate if needed:
- edit `page_XX.spoken.json -> paragraphs[].spoken_text` if wording is wrong
- regenerate by paragraph index if wording is correct but audio sounds bad

```text
[3/5] Voice Generation
Paragraph indices to generate (comma separated, empty means all): 4,7
Optional volume gain for this regeneration (e.g. 0.9, 1.1, default empty): 1.1

Stage 3 completed.
manifest: outputs/<voice_name>/<page_name>/segments_manifest.json
segments_dir: outputs/<voice_name>/<page_name>/segments
Available paragraphs:
2, 3, 4, 5, 6, 7

Stage 3 review action
- c: continue to the next stage
- r: regenerate one or more paragraphs
- s: stop here
Choice (c/r/s) [c]: r
Enter paragraph indices to regenerate (comma separated or 'all'): 4,7
```

### Stage 4. Timeline Alignment

**Output and Review**

Check:
- paragraph starts
- ordering issues

If a cover intro is enabled, the cover paragraph does not need a body timeline start.
For example, when `cover_paragraph_index=2`, paragraph 2 is treated as the intro segment and the body timeline starts from paragraph 3.

Edit if needed:
- `page_XX.timeline.final.json -> segments[].start`

The timeline stage now reads script timestamps directly. If a non-cover narration paragraph is missing a timestamp, Stage 4 fails and tells you which paragraph needs a `0.0s,`-style prefix.

```text
[4/5] Timeline Alignment
Stage 4 completed.
timeline: outputs/scripts/<page_name>/page_05.timeline.final.json
status: complete
missing: []

Stage 4 review action
- c: continue to the next stage
- s: stop here
Choice (c/s) [c]: c
```

### Stage 5. Video Composition

**Output and Review**

Check:
- final pacing
- retiming quality
- audio-video alignment

If a cover intro is enabled, Stage 5 will:
- prepend the cover image as a static intro clip
- use the selected cover paragraph audio at the beginning
- shift the main video body after the intro duration

If an outro page is enabled, Stage 5 will:
- append a static outro page after the main video
- use a fixed slogan audio if provided
- otherwise generate the slogan audio from the current voice profile and append it at the end

If something is wrong:
- go back to Stage 3 for audio issues
- go back to Stage 4 for timing issues

```text
[5/5] Video Composition
Stage 5 completed.
final_video: outputs/composed/<page_name>/page_composed.mp4
output_dir:   outputs/composed/<page_name>
```

## Current Behavior

- paragraph-level audio generation
- optional paragraph selection and volume gain during Stage 3 voice generation
- profile path accepts either a `.pt` file or a profile directory containing `<dirname>.pt`
- text extraction filters some short header/footer-like slide text blocks based on layout position
- optional cover intro support with `cover_image` and `cover_paragraph_index`
- when a cover intro is enabled, the cover paragraph is excluded from body timeline alignment and the body starts from the next paragraph
- optional outro page support with either fixed slogan audio or generated slogan audio
- start-driven timeline semantics
- no API-key dependency for timeline generation
- local video retiming instead of truncating audio
- current composition defaults:
  - `buffer_sec = 1.2`
  - `tail_buffer_sec = 1.5`
  - `audio_tail_pad_sec = 0.5`

## Limitations

- some source PPT files may contain malformed text encoding
- every non-cover narration paragraph must provide a timestamp prefix before timeline generation
- human review is still recommended for production-quality output
- some sentence endings may require spoken-text rewriting for better TTS delivery
- the current workflow is page-oriented rather than a full-deck production pipeline

## Project Structure

- `outputs/voice_profiles/`: saved voice profiles `.pt`
- `outputs/<voice_name>/<title>/`: generated voice artifacts for a page
- `outputs/scripts/`: extracted text, spoken text, and timeline files
- `pipeline/`: reusable page-level workflow scripts and older tooling
- `sample/`: reference examples or experiments

## Roadmap

- keep simplifying `run_pipeline.py` interaction without making the workflow more rigid
- improve README examples with more realistic sample commands and outputs
- add better review and recovery tools for paragraph-level regeneration and timeline fixing
