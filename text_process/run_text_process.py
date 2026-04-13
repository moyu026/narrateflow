from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
SCRIPTS_DIR = OUTPUTS_DIR / "scripts"
DEFAULT_RULES_PATH = (
    Path(__file__).resolve().parent / "config" / "pronunciation_rules.json"
)
PPT_NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}


def slugify(text: str, max_len: int = 60) -> str:
    text = re.sub(r"[\\/:*?\"<>|]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("：", " ").replace(":", " ")
    text = re.sub(r"\s+", "", text)
    return (text[:max_len] or "untitled").strip() or "untitled"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def create_script_paths(ppt_path: Path, page: int) -> tuple[Path, Path]:
    title_dir = get_ppt_output_dir_name(ppt_path, page)
    out_dir = SCRIPTS_DIR / title_dir
    return (
        out_dir / f"page_{page:02d}.extracted.json",
        out_dir / f"page_{page:02d}.spoken.json",
    )


def get_ppt_output_dir_name(ppt_path: Path, page: int) -> str:
    try:
        paragraphs = extract_slide_paragraphs(ppt_path, page)
    except Exception:
        paragraphs = []
    title = paragraphs[0]["text"] if paragraphs else ppt_path.stem
    return slugify(title, max_len=60)


def load_pronunciation_rules(rules_path: str | None = None) -> list[dict[str, Any]]:
    path = Path(rules_path) if rules_path else DEFAULT_RULES_PATH
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("rules", [])
    if not isinstance(payload, list):
        raise ValueError(
            "发音规则文件格式不正确，应为 JSON 数组或带 rules 字段的对象。"
        )
    return [rule for rule in payload if isinstance(rule, dict) and rule.get("pattern")]


def apply_pronunciation_rules(
    text: str, rules: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    current = text
    applied: list[dict[str, Any]] = []
    for rule in rules:
        pattern = str(rule["pattern"])
        replacement = str(rule.get("replacement", ""))
        ignore_case = bool(rule.get("ignore_case", False))
        is_regex = bool(rule.get("regex", False))
        flags = re.IGNORECASE if ignore_case else 0

        if is_regex:
            updated, count = re.subn(pattern, replacement, current, flags=flags)
        elif ignore_case:
            updated, count = re.subn(
                re.escape(pattern), replacement, current, flags=flags
            )
        else:
            count = current.count(pattern)
            updated = current.replace(pattern, replacement)

        if count:
            applied.append(
                {
                    "pattern": pattern,
                    "replacement": replacement,
                    "count": count,
                    "note": rule.get("note", ""),
                }
            )
        current = updated
    return current, applied


def normalize_spoken_text(text: str) -> str:
    text = text.replace("（", "，").replace("）", "")
    text = text.replace("(", "，").replace(")", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([，。！？；：,.!?;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def finalize_segment_text(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if re.search(r"[，,、；;：:]$", text):
        return text[:-1].rstrip() + "。"
    if re.search(r"[。！？!?]$", text):
        return text
    return text + "。"


def chunk_text_by_length(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    comma_parts = re.split(r"([，,、；;：:])", text)
    combined_parts: list[str] = []
    for i in range(0, len(comma_parts), 2):
        frag = comma_parts[i]
        delim = comma_parts[i + 1] if i + 1 < len(comma_parts) else ""
        item = (frag + delim).strip()
        if item:
            combined_parts.append(item)

    chunks: list[str] = []
    current = ""
    for part in combined_parts:
        if not current:
            current = part
        elif len(current) + 1 + len(part) <= max_chars:
            current = f"{current} {part}"
        elif re.search(r"[，,、；;：:]$", current) and len(current) + 1 + len(
            part
        ) <= int(max_chars * 1.35):
            current = f"{current} {part}"
        else:
            chunks.append(current.strip())
            current = part
    if current:
        chunks.append(current.strip())

    final_chunks: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
            continue

        words = chunk.split(" ")
        if len(words) > 1:
            current_word_chunk = ""
            for word in words:
                if not current_word_chunk:
                    current_word_chunk = word
                elif len(current_word_chunk) + 1 + len(word) <= max_chars:
                    current_word_chunk = f"{current_word_chunk} {word}"
                else:
                    final_chunks.append(current_word_chunk.strip())
                    current_word_chunk = word
            if current_word_chunk:
                final_chunks.append(current_word_chunk.strip())
            continue

        for start in range(0, len(chunk), max_chars):
            final_chunks.append(chunk[start : start + max_chars].strip())

    return [chunk for chunk in final_chunks if chunk]


def split_spoken_paragraph(text: str, max_chars: int) -> list[str]:
    text = normalize_spoken_text(text)
    if not text:
        return []

    if len(text) <= max_chars:
        final = finalize_segment_text(text)
        return [final] if final else []

    strong_parts = [
        part.strip() for part in re.split(r"(?<=[。！？!?])\s*", text) if part.strip()
    ]
    if not strong_parts:
        strong_parts = [text]

    if len(strong_parts) == 1 and len(text) <= int(max_chars * 1.45):
        final = finalize_segment_text(text)
        return [final] if final else []

    segments: list[str] = []
    for part in strong_parts:
        for chunk in chunk_text_by_length(part, max_chars=max_chars):
            final = finalize_segment_text(chunk)
            if final:
                segments.append(final)
    return segments


def extract_slide_paragraphs(ppt_path: Path, page: int) -> list[dict[str, Any]]:
    slide_name = f"ppt/slides/slide{page}.xml"
    with zipfile.ZipFile(ppt_path) as archive:
        if slide_name not in archive.namelist():
            raise FileNotFoundError(f"PPT 中不存在第 {page} 页。")
        root = ET.fromstring(archive.read(slide_name))

    paragraphs: list[dict[str, Any]] = []
    para_index = 1

    def para_text(para: ET.Element) -> str:
        parts: list[str] = []
        for child in list(para):
            tag = child.tag.split("}")[-1]
            if tag == "r":
                node = child.find("a:t", PPT_NS)
                if node is not None and node.text:
                    parts.append(node.text)
            elif tag == "fld":
                node = child.find("a:t", PPT_NS)
                if node is not None and node.text:
                    parts.append(node.text)
            elif tag == "br":
                parts.append("\n")
        return "".join(parts).strip()

    for shape in root.findall(
        ".//p:sp",
        {**PPT_NS, "p": "http://schemas.openxmlformats.org/presentationml/2006/main"},
    ):
        tx_body = shape.find(
            "p:txBody",
            {
                **PPT_NS,
                "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
            },
        )
        if tx_body is None:
            continue
        for para in tx_body.findall("a:p", PPT_NS):
            text = para_text(para)
            if text:
                paragraphs.append({"index": para_index, "text": text})
                para_index += 1
    return paragraphs


def prepare_ppt_page(
    ppt_path: Path,
    page: int,
    rules_path: str | None = None,
    max_chars: int = 72,
) -> dict[str, Any]:
    paragraphs = extract_slide_paragraphs(ppt_path, page)
    rules = load_pronunciation_rules(rules_path)
    extracted_path, spoken_path = create_script_paths(ppt_path, page)
    title_text = paragraphs[0]["text"] if paragraphs else ""

    extracted_payload = {
        "ppt_path": str(ppt_path.resolve()),
        "page": page,
        "paragraph_count": len(paragraphs),
        "paragraphs": paragraphs,
        "combined_text": "\n".join(item["text"] for item in paragraphs),
    }

    spoken_paragraphs: list[dict[str, Any]] = []
    spoken_segments: list[dict[str, Any]] = []
    applied_rules: list[dict[str, Any]] = []
    segment_index = 1

    for paragraph in paragraphs:
        spoken_text, paragraph_rules = apply_pronunciation_rules(
            paragraph["text"], rules
        )
        spoken_text = normalize_spoken_text(spoken_text)
        is_title = paragraph["index"] == 1
        segments = (
            [] if is_title else split_spoken_paragraph(spoken_text, max_chars=max_chars)
        )
        spoken_paragraphs.append(
            {
                "index": paragraph["index"],
                "is_title": is_title,
                "source_text": paragraph["text"],
                "spoken_text": spoken_text,
                "segment_count": len(segments),
                "applied_rules": paragraph_rules,
            }
        )
        if paragraph_rules:
            applied_rules.extend(
                [
                    {"paragraph_index": paragraph["index"], **rule}
                    for rule in paragraph_rules
                ]
            )

        for segment_text in segments:
            spoken_segments.append(
                {
                    "segment_id": f"p{page:02d}_s{segment_index:03d}",
                    "paragraph_index": paragraph["index"],
                    "source_text": paragraph["text"],
                    "spoken_text": segment_text,
                }
            )
            segment_index += 1

    spoken_payload = {
        "ppt_path": str(ppt_path.resolve()),
        "page": page,
        "title_text": title_text,
        "rules_path": str(
            (Path(rules_path) if rules_path else DEFAULT_RULES_PATH).resolve()
        )
        if (Path(rules_path) if rules_path else DEFAULT_RULES_PATH).exists()
        else None,
        "max_chars": max_chars,
        "paragraphs": spoken_paragraphs,
        "segments": spoken_segments,
        "applied_rules": applied_rules,
        "combined_spoken_text": " ".join(
            item["spoken_text"] for item in spoken_segments
        ),
    }

    write_json(extracted_path, extracted_payload)
    write_json(spoken_path, spoken_payload)
    return {
        "extracted_path": extracted_path,
        "spoken_path": spoken_path,
        "extracted": extracted_payload,
        "spoken": spoken_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 text processing")
    parser.add_argument("--ppt", required=True)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--rules", default=None)
    parser.add_argument("--max-chars", type=int, default=72)
    args = parser.parse_args()

    result = prepare_ppt_page(
        ppt_path=Path(args.ppt),
        page=args.page,
        rules_path=args.rules,
        max_chars=args.max_chars,
    )
    print(
        json.dumps(
            {
                "extracted_path": str(result["extracted_path"]),
                "spoken_path": str(result["spoken_path"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
