# NarrateFlow

NarrateFlow is a human-in-the-loop pipeline for turning PPTs or documents into dubbed videos.

It extracts narration text, generates paragraph-level voice audio, aligns audio to a target video timeline with a vision-language model, and renders the final dubbed video.

## Environment

| Component | Requirement | Notes |
|---|---|---|
| Python | 3.13 | Recommended runtime |
| FFmpeg / FFprobe | Required | Used for frame extraction and video composition |
| CUDA | Recommended | Speeds up local TTS inference |
| Local TTS backend | Qwen-TTS | Used in `voice_process` |
| VL backend | Qwen2.5-VL-72B via MAAS API | Used in `timeline_align` |
| API key | `MAAS_API_KEY` | Required for timeline alignment |

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

The recommended way to run NarrateFlow is through the interactive pipeline entrypoint:

```bash
python run_pipeline.py
```

The pipeline will:

1. ask for missing inputs interactively
2. run the stages in order
3. pause after reviewable stages
4. allow paragraph-level regeneration during voice review

## Example Interactive Flow

Below is a simplified example of what an interactive run looks like.

### Input Collection

```text
PPT path: D:\work\example.pptx
Page number: 5
Target video path: D:\work\example.mp4

Title mode
- first: treat the first paragraph as title
- none: treat all paragraphs as narration
- manual: choose title paragraph indices manually
Choice (first/none/manual) [first]: first

Do you already have a voice profile file (y/n) [y]: y
Voice profile path (.pt file or profile directory): D:\qwen3-tts\outputs\voice_profiles\reference_voice

Initial probe times (comma separated, keyframe times) [0,10,20,30]: 0,10,20,30
```

### Stage 1. Text Processing

**Output and Review**

Check:
- paragraph extraction
- title handling
- spoken narration wording

Edit if needed:
- `page_XX.spoken.json -> paragraphs[].spoken_text`

```text
[1/5] Text Processing
Stage 1 completed.
extracted_json: D:\...\page_05.extracted.json
spoken_json:    D:\...\page_05.spoken.json

Stage 1 review action
- c: continue to the next stage
- s: stop here
Choice (c/s) [c]: c
```

### Stage 2. Voice Profile Generation

**Output**

```text
[2/5] Voice Profile Generation (skipped, using existing profile)
profile_path: D:\...\reference_voice.pt
```

### Stage 3. Voice Generation

**Output and Review**

Check:
- paragraph-level audio quality
- omitted or weakly spoken words
- sentence endings

Edit or regenerate if needed:
- edit `page_XX.spoken.json -> paragraphs[].spoken_text` if wording is wrong
- regenerate by paragraph index if wording is correct but audio sounds bad

```text
[3/5] Voice Generation
Stage 3 completed.
manifest: D:\...\segments_manifest.json
segments_dir: D:\...\segments
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
- missing paragraphs
- ordering issues

Edit if needed:
- `page_XX.timeline.final.json -> segments[].start`
- for missing paragraphs, set `matched=true` and provide `start`

```text
[4/5] Timeline Alignment
Stage 4 completed.
timeline: D:\...\page_05.timeline.final.json
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

If something is wrong:
- go back to Stage 3 for audio issues
- go back to Stage 4 for timing issues

```text
[5/5] Video Composition
Stage 5 completed.
final_video: D:\...\page_composed.mp4
output_dir:   D:\...\outputs\composed\page_xx
```

## Current Behavior

- paragraph-level audio generation
- start-driven timeline semantics
- local video retiming instead of truncating audio
- current composition defaults:
  - `buffer_sec = 1.2`
  - `tail_buffer_sec = 1.5`
  - `audio_tail_pad_sec = 0.5`

## Limitations

- some source PPT files may contain malformed text encoding
- timeline alignment quality depends on UI visibility, subtitle availability, and visual distinction between adjacent segments
- human review is still recommended for production-quality output
- some sentence endings may require spoken-text rewriting for better TTS delivery

## Roadmap

- support more document input formats
- improve keyframe selection and OCR-assisted timeline alignment
- improve adjacent-paragraph conflict resolution in timeline generation
- add paragraph-level ASR review and regeneration workflow
- refine the interactive pipeline experience
