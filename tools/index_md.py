"""Chunk and export markdown content for embedding.

Usage:
  python -m tools.index_md --input TABLE\ Chatbot.md --out chunks.jsonl --chunk-size 800

This will split the markdown into logical sections (by headings) and also into text windows
of approximately `chunk_size` characters, then write JSONL with fields: id, title, text.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from pathlib import Path
from typing import List


def split_markdown_into_sections(md: str) -> List[dict]:
    # Split by top-level headings (lines starting with #) but keep content when no headings.
    lines = md.splitlines()
    sections = []
    cur_title = "document"
    cur_lines: List[str] = []

    heading_re = re.compile(r"^#{1,6}\s+(.*)")

    for line in lines:
        m = heading_re.match(line)
        if m:
            # flush previous
            if cur_lines:
                sections.append({"title": cur_title.strip(), "text": "\n".join(cur_lines).strip()})
            cur_title = m.group(1)
            cur_lines = []
        else:
            cur_lines.append(line)

    if cur_lines:
        sections.append({"title": cur_title.strip(), "text": "\n".join(cur_lines).strip()})

    # If no section titles, collapse to one
    if not sections:
        sections = [{"title": "document", "text": md}]

    return sections


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end == len(text):
            break
        start = end - overlap
    return chunks


def build_chunks_from_file(path: Path, chunk_size: int = 800) -> List[dict]:
    md = path.read_text(encoding="utf-8")
    sections = split_markdown_into_sections(md)
    items = []
    for sec in sections:
        title = sec.get("title") or "section"
        text = sec.get("text") or ""
        for i, c in enumerate(chunk_text(text, chunk_size=chunk_size)):
            items.append({
                "id": str(uuid.uuid4()),
                "title": title + (f" [{i}]" if i else ""),
                "text": c,
            })
    return items


def write_jsonl(items: List[dict], out: Path) -> None:
    with out.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="input markdown file path")
    p.add_argument("--out", required=True, help="output jsonl path")
    p.add_argument("--chunk-size", type=int, default=800)
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    items = build_chunks_from_file(input_path, chunk_size=args.chunk_size)
    out_path = Path(args.out)
    write_jsonl(items, out_path)
    print(f"Wrote {len(items)} chunks to {out_path}")


if __name__ == "__main__":
    main()
