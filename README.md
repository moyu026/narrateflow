# NarrateFlow

NarrateFlow is a multimodal pipeline for turning PPTs or documents into dubbed videos.

It extracts spoken narration text, generates paragraph-level voice audio with voice cloning, aligns audio to a target video timeline using keyframes and a vision-language model, and renders the final dubbed video.

## Features

- PPT/document text extraction
- spoken-script normalization and terminology replacement
- voice profile generation from reference audio
- paragraph-level audio generation
- keyframe-based timeline alignment
- vision-language-model-assisted timeline matching
- local video retiming when audio is longer than the available video segment
- human-in-the-loop review workflow for text, audio, timeline, and final video

## Pipeline

NarrateFlow is organized into 4 stages:

1. Text Processing
2. Voice Processing
3. Timeline Alignment
4. Video Composition

## Project Structure

```text
text_process/
  run_text_process.py
  config/
    pronunciation_rules.json

voice_process/
  common.py
  run_voice_profile.py
  run_voice_generate.py

timeline_align/
  run_timeline_align.py
  keyframe_filter.py
  vl_client.py

video_compose/
  run_video_compose.py
```

## Inputs

Current main inputs include:

- `.pptx` files
- reference audio and reference text for voice profile creation
- saved voice profile files
- target video files

## Outputs

### Stage 1
- `page_XX.extracted.json`
- `page_XX.spoken.json`

### Stage 2
- `segments_manifest.json`
- `segments/*.wav`

### Stage 3
- `page_XX.timeline.final.json`
- `page_XX.timeline.final.json.debug.json`

### Stage 4
- `page_audio.wav`
- `page_retimed_video.mp4`
- `page_composed.mp4`
- `page_plan.json`

## Workflow

### Stage 1: Text Processing

```bash
python text_process/run_text_process.py --ppt "inputs/example.pptx" --page 1
```

### Stage 2A: Voice Profile Generation

```bash
python voice_process/run_voice_profile.py \
  --voice-name selina \
  --ref-audio "path/to/ref.wav" \
  --ref-text "reference text"
```

### Stage 2B: Voice Generation

```bash
python voice_process/run_voice_generate.py \
  --spoken-json "outputs/scripts/<page_title>/page_01.spoken.json" \
  --profile "outputs/voice_profiles/selina/selina.pt" \
  --voice-name selina
```

Single-paragraph regeneration is also supported:

```bash
python voice_process/run_voice_generate.py \
  --spoken-json "outputs/scripts/<page_title>/page_01.spoken.json" \
  --profile "outputs/voice_profiles/selina/selina.pt" \
  --voice-name selina \
  --paragraph-index 4
```

### Stage 3: Timeline Alignment

```bash
python timeline_align/run_timeline_align.py \
  --video "path/to/video.mp4" \
  --spoken-json "outputs/scripts/<page_title>/page_01.spoken.json" \
  --output "outputs/scripts/<page_title>/page_01.timeline.final.json" \
  --probe-mode keyframes \
  --probe-times "0,10,20,30"
```

### Stage 4: Video Composition

```bash
python video_compose/run_video_compose.py \
  --mode retime \
  --video "path/to/video.mp4" \
  --timeline "outputs/scripts/<page_title>/page_01.timeline.final.json" \
  --segments-manifest "outputs/<voice_name>/<page_title>/segments_manifest.json" \
  --output-dir "outputs/composed/page_01"
```

## Human Review

NarrateFlow is designed as a human-in-the-loop system.

Recommended review points:

### After Stage 1
Check:
- extracted text correctness
- title/body separation
- terminology replacement
- spoken-text readability

### After Stage 2
Check:
- paragraph-level audio quality
- omitted or weakly spoken words
- sentence ending quality
- whether any paragraph needs regeneration

### After Stage 3
Check:
- paragraph `start` positions
- missing paragraphs
- timeline order and overlap issues

### After Stage 4
Check:
- final pacing
- retimed video smoothness
- whether audio-video alignment is acceptable

## Timeline Semantics

NarrateFlow uses `start` as the primary timeline anchor.

- `start`: the main audio insertion point
- `end_hint`: a reference window only, not a hard trimming boundary

Actual playback duration is decided in Stage 4 using:
- audio duration
- buffer
- next segment start
- local video retiming when necessary

## Current Defaults

Stage 4 currently uses these default values:

- `buffer_sec = 1.2`
- `tail_buffer_sec = 1.5`
- `audio_tail_pad_sec = 0.5`

## Supported Inputs

Currently supported:
- `.pptx`

Planned support:
- `.docx`
- `.md`
- `.txt`
- direct plain text input

## Limitations

- Some source PPT files may contain malformed text encoding.
- Timeline alignment quality depends on UI visibility, subtitle availability, and visual distinction between adjacent segments.
- Human review is still recommended for high-quality production use.
- Some TTS sentence endings may require spoken-text rewriting for more natural delivery.

## Environment

Recommended environment:

- Python 3.13
- ffmpeg / ffprobe
- CUDA-capable environment for faster TTS
- MAAS API key for Stage 3

## Notes

- `outputs/` contains generated artifacts and should usually not be committed.
- `debug/` artifacts are useful for troubleshooting but are not meant as final delivery files.
- The current design prioritizes reviewability and controllability over full automation.

## Roadmap

- support more document input formats
- improve OCR + VL combined alignment
- improve adjacent-paragraph conflict resolution in timeline generation
- improve paragraph-level ASR review and regeneration workflow
- provide a cleaner end-to-end CLI entrypoint

## License

TBD
