import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from PIL import Image
from transformers import AutoModel, AutoTokenizer


MODEL_PATH = r"D:/qwen3-tts/models/OpenBMB/MiniCPM-V-4-int4"


def load_candidates(spoken_json: Path, paragraph_indices: list[int]) -> dict[int, str]:
    payload = json.loads(spoken_json.read_text(encoding="utf-8"))
    mapping = {}
    for item in payload.get("paragraphs", []):
        idx = int(item["index"])
        if idx in paragraph_indices:
            mapping[idx] = item["spoken_text"]
    return mapping


def extract_window_frames(
    video_path: Path, out_dir: Path, start_sec: int, offsets=(0, 2, 4)
):
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, off in enumerate(offsets, start=1):
        out_file = out_dir / f"f{i}.png"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(start_sec + off),
                "-i",
                str(video_path).replace("\\", "/"),
                "-update",
                "1",
                "-frames:v",
                "1",
                str(out_file).replace("\\", "/"),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        frames.append(Image.open(out_file).convert("RGB"))
    return frames


def build_question(candidates: dict[int, str]) -> str:
    lines = [
        "这三张图是同一段视频窗口内的连续画面。请只从候选段落里选出最匹配的一个。",
        "如果看不出来，请返回 unknown。",
        "请严格只输出一行 JSON，不要输出任何额外解释：",
        '{"best_paragraph_index": 2, "reason": "..."}',
        "候选段落：",
    ]
    for idx, text in candidates.items():
        lines.append(f"- {idx}: {text}")
    return "\n".join(lines)


def parse_answer(text: str, candidates: dict[int, str]) -> dict:
    try:
        start = text.rfind("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
    except Exception:
        pass
    patterns = [
        r"best_paragraph_index\D+(\d+)",
        r"paragraph is(?: the)? (\w+)",
        r'"(\d+):',
        r"最匹配的段落是(?:第)?(\d+)段",
    ]
    mapping = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
        "ninth": 9,
        "tenth": 10,
    }
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            value = m.group(1)
            if value.lower() in mapping:
                return {"best_paragraph_index": mapping[value.lower()], "reason": text}
            if value.isdigit():
                return {"best_paragraph_index": int(value), "reason": text}
    for idx, candidate_text in candidates.items():
        short = candidate_text.strip().replace(" ", "")
        if short and short[:20] in text.replace(" ", ""):
            return {"best_paragraph_index": idx, "reason": text}
    return {"best_paragraph_index": None, "reason": text}


def merge_windows(results: list[dict]) -> list[dict]:
    merged = []
    current = None
    for item in results:
        idx = item.get("best_paragraph_index")
        if current is None or current["best_paragraph_index"] != idx:
            if current is not None:
                merged.append(current)
            current = {
                "best_paragraph_index": idx,
                "start": item["window_start"],
                "end": item["window_end"],
                "reasons": [item.get("reason", "")],
            }
        else:
            current["end"] = item["window_end"]
            current["reasons"].append(item.get("reason", ""))
    if current is not None:
        merged.append(current)
    return merged


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="MiniCPM window-level timeline matching prototype"
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--spoken-json", required=True)
    parser.add_argument(
        "--paragraphs", required=True, help="comma separated paragraph indices"
    )
    parser.add_argument("--start-sec", type=int, default=0)
    parser.add_argument("--end-sec", type=int, default=30)
    parser.add_argument("--step-sec", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paragraph_indices = [
        int(x.strip()) for x in args.paragraphs.split(",") if x.strip()
    ]
    candidates = load_candidates(Path(args.spoken_json), paragraph_indices)

    print("Loading MiniCPM-V-4-int4 on CPU...")
    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, dtype="auto", attn_implementation="sdpa"
    )
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    total_start = time.perf_counter()
    results = []
    for start in range(args.start_sec, args.end_sec, args.step_sec):
        print(f"Matching window {start}-{start + 4}s...")
        frames = extract_window_frames(
            Path(args.video), Path(r"D:/qwen3-tts/tmp") / f"mw_{start:03d}", start
        )
        question = build_question(candidates)
        msgs = [{"role": "user", "content": frames + [question]}]
        infer_start = time.perf_counter()
        answer = model.chat(msgs=msgs, tokenizer=tokenizer, max_new_tokens=256)
        infer_sec = round(time.perf_counter() - infer_start, 3)
        parsed = parse_answer(answer, candidates)
        results.append(
            {
                "window_start": start,
                "window_end": start + 4,
                "infer_sec": infer_sec,
                "raw_answer": answer,
                "best_paragraph_index": parsed.get("best_paragraph_index"),
                "reason": parsed.get("reason", ""),
            }
        )

    total_sec = round(time.perf_counter() - total_start, 3)
    avg_sec = (
        round(sum(item["infer_sec"] for item in results) / len(results), 3)
        if results
        else 0.0
    )

    payload = {
        "video": args.video,
        "spoken_json": args.spoken_json,
        "paragraphs": paragraph_indices,
        "total_infer_sec": total_sec,
        "avg_window_infer_sec": avg_sec,
        "results": results,
        "merged_windows": merge_windows(results),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
