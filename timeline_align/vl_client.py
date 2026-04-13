from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import time
from pathlib import Path

import requests


API_URL = "https://api.modelarts-maas.com/v1/chat/completions"
REQUEST_INTERVAL_SEC = 1.0
MAX_RETRIES = 2
RETRY_BACKOFF_SEC = 3.0


def load_api_key(explicit_key: str | None = None) -> str:
    api_key = explicit_key or os.environ.get("MAAS_API_KEY")
    if not api_key:
        raise SystemExit("环境变量 MAAS_API_KEY 未设置，当前进程无法读取。")
    return api_key


def load_page_segments(spoken_json: Path) -> tuple[str, list[dict]]:
    payload = json.loads(spoken_json.read_text(encoding="utf-8"))
    return payload.get("title_text", ""), payload.get("segments", [])


def build_prompt(
    title: str, segments: list[dict], frame_hint: str | None = None
) -> str:
    lines = [
        "你正在帮助做PPT视频配音时间轴对齐。",
        "请只根据这张视频帧图片，判断当前画面最可能对应下面哪一段讲稿。",
        "如果看不清字幕或无法判断，请明确返回 unknown。",
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
                "reason": "一句简短原因，如果无法判断就说明原因",
            },
            ensure_ascii=False,
        )
    )
    return "\n".join(lines)


def call_vl(api_key: str, image_path: Path, prompt: str) -> dict:
    image_mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    base64_image = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    payload = {
        "model": "qwen2.5-vl-72b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_mime};base64,{base64_image}"
                        },
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
                API_URL,
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
    raise RuntimeError("VL request failed unexpectedly")


def safe_parse_json_from_content(content: str) -> dict:
    content = content.strip()
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", content, flags=re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass
    bare = re.search(r"(\{.*\})", content, flags=re.S)
    if bare:
        try:
            return json.loads(bare.group(1))
        except Exception:
            pass
    return {
        "title_match": False,
        "subtitle_text": "",
        "best_paragraph_index": None,
        "confidence": 0.0,
        "reason": "模型返回无法解析为标准 JSON，已按 unknown 处理。",
        "parse_error": True,
        "raw_content": content,
    }


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
    for frame in frames:
        hint = (
            frame_hint_builder(frame)
            if frame_hint_builder
            else f"当前视频时间点约为 {frame['time']} 秒。请注意允许返回 unknown。"
        )
        prompt = build_prompt(title=title, segments=segments, frame_hint=hint)
        response = call_vl(
            api_key=api_key, image_path=Path(frame["image_path"]), prompt=prompt
        )
        content = response["choices"][0]["message"]["content"]
        parsed = safe_parse_json_from_content(content)
        parsed["time"] = frame["time"]
        parsed["image_path"] = frame["image_path"]
        results.append(parsed)
    return results
