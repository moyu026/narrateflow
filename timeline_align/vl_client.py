from __future__ import annotations

import base64
import io
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import requests
from tqdm.auto import tqdm


QWEN_API_URL = "https://api.modelarts-maas.com/v1/chat/completions"
REQUEST_INTERVAL_SEC = 1.0
MAX_RETRIES = 2
RETRY_BACKOFF_SEC = 3.0


def load_api_key(explicit_key: str | None = None) -> str:
    api_key = explicit_key or os.environ.get("MAAS_API_KEY")
    if not api_key:
        raise SystemExit("Environment variable MAAS_API_KEY is required.")
    return api_key


def load_gemini_api_key(explicit_key: str | None = None) -> str:
    api_key = explicit_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Environment variable GEMINI_API_KEY is required.")
    return api_key


def load_page_segments(spoken_json: Path) -> tuple[str, list[dict]]:
    payload = json.loads(spoken_json.read_text(encoding="utf-8"))
    return payload.get("title_text", ""), payload.get("segments", [])


def build_prompt(
    title: str, segments: list[dict], frame_hint: str | None = None
) -> str:
    lines = [
        "你正在帮助做视频讲解段落与时间轴对齐。",
        "请仅根据当前图片判断它最可能对应哪个候选段落。",
        "如果无法判断，请明确返回 unknown。",
    ]
    if frame_hint:
        lines.append(f"补充信息: {frame_hint}")
    lines.append(f"页面标题: {title}")
    lines.append("候选段落如下:")
    for segment in segments:
        lines.append(
            f"- paragraph_index={segment['paragraph_index']}: {segment['spoken_text']}"
        )
    lines.append("请严格返回 JSON，不要输出额外解释。")
    lines.append(
        json.dumps(
            {
                "title_match": True,
                "subtitle_text": "从画面中能读到的字幕文本，没有就写空字符串",
                "best_paragraph_index": 2,
                "confidence": 0.0,
                "reason": "一句简短原因，无法判断时说明原因",
            },
            ensure_ascii=False,
        )
    )
    return "\n".join(lines)


def _encode_image_as_data_url(image_path: Path) -> str:
    image_mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    base64_image = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{image_mime};base64,{base64_image}"


def call_vl_qwen(api_key: str, image_path: Path, prompt: str) -> dict:
    payload = {
        "model": "qwen2.5-vl-72b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": _encode_image_as_data_url(image_path)},
                    },
                ],
            }
        ],
        "temperature": 0.1,
    }
    session = requests.Session()
    session.trust_env = False
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = session.post(
                QWEN_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                timeout=180,
                verify=False,
            )
            response.raise_for_status()
            if REQUEST_INTERVAL_SEC > 0:
                time.sleep(REQUEST_INTERVAL_SEC)
            return response.json()
        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
        ) as exc:
            last_error = exc
            if attempt >= MAX_RETRIES:
                raise
            time.sleep(RETRY_BACKOFF_SEC * (attempt + 1))
    if last_error:
        raise last_error
    raise RuntimeError("Qwen VL request failed unexpectedly.")


def _annotate_frame_bytes(
    image_path: Path, label: str, time_text: str | None = None
) -> bytes:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    banner_h = max(28, min(56, frame.shape[0] // 10))
    cv2.rectangle(frame, (0, 0), (min(frame.shape[1], 240), banner_h), (0, 0, 0), -1)
    text = label if not time_text else f"{label} {time_text}"
    cv2.putText(
        frame,
        text,
        (8, max(18, banner_h - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError(f"Unable to encode annotated image: {image_path}")
    return encoded.tobytes()


def call_vl_gemini(
    api_key: str,
    windows: list[dict[str, Any]],
    prompt: str,
    model: str = "gemini-2.5-pro",
) -> dict:
    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(model)

    contents: list[Any] = [prompt]
    for window in windows:
        contents.append(
            f"Window {window['window_id']} | {window['start_time']:.2f}s - {window['end_time']:.2f}s"
        )
        for index, frame in enumerate(window.get("frames", []), start=1):
            frame_label = f"#{index}"
            time_text = f"@ {float(frame['time']):.2f}s"
            image_bytes = _annotate_frame_bytes(Path(frame["image_path"]), frame_label, time_text)
            contents.append(Image.open(io.BytesIO(image_bytes)))

    response = client.generate_content(contents)
    return {"content": getattr(response, "text", "")}


def _extract_json_snippet(content: str) -> str | None:
    content = content.strip()
    fenced = re.search(r"```json\s*(\[.*?\]|\{.*?\})\s*```", content, flags=re.S)
    if fenced:
        return fenced.group(1)
    bare = re.search(r"(\[.*\]|\{.*\})", content, flags=re.S)
    if bare:
        return bare.group(1)
    return None


def safe_parse_json_from_content(content: str) -> dict:
    snippet = _extract_json_snippet(content)
    if snippet:
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {
        "title_match": False,
        "subtitle_text": "",
        "best_paragraph_index": None,
        "confidence": 0.0,
        "reason": "模型返回内容无法解析为标准 JSON，已按 unknown 处理。",
        "parse_error": True,
        "raw_content": content,
    }


def parse_gemini_batch_response(content: str) -> list[dict[str, Any]]:
    snippet = _extract_json_snippet(content)
    if snippet:
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
            if isinstance(parsed, dict):
                items = parsed.get("windows")
                if isinstance(items, list):
                    return [item for item in items if isinstance(item, dict)]
        except Exception:
            pass
    return []


def extract_frames_at_times(
    video_path: Path, out_dir: Path, times: list[float]
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for t in times:
        stem = f"frame_{t:06.2f}".replace(".", "_")
        image_path = out_dir / f"{stem}.png"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{t}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-update",
                "1",
                str(image_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        frames.append({"time": round(t, 2), "image_path": str(image_path)})
    return frames


def load_selected_keyframes(
    keyframe_json: Path, selected_times: list[float]
) -> list[dict]:
    payload = json.loads(keyframe_json.read_text(encoding="utf-8"))
    rounded = {round(t, 2) for t in selected_times}
    return sorted(
        [
            item
            for item in payload.get("candidates", [])
            if round(float(item["time"]), 2) in rounded
        ],
        key=lambda item: item["time"],
    )


def probe_frames(
    api_key: str,
    title: str,
    segments: list[dict],
    frames: list[dict],
    frame_hint_builder=None,
) -> list[dict]:
    results = []
    for frame in tqdm(
        frames,
        total=len(frames),
        desc="Processing keyframes with VLM",
        unit="frame",
        leave=False,
    ):
        hint = (
            frame_hint_builder(frame)
            if frame_hint_builder
            else f"当前视频时间点约为 {frame['time']} 秒。请注意允许返回 unknown。"
        )
        prompt = build_prompt(title=title, segments=segments, frame_hint=hint)
        response = call_vl_qwen(
            api_key=api_key, image_path=Path(frame["image_path"]), prompt=prompt
        )
        content = response["choices"][0]["message"]["content"]
        parsed = safe_parse_json_from_content(content)
        parsed["time"] = frame["time"]
        parsed["image_path"] = frame["image_path"]
        results.append(parsed)
    return results
