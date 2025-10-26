"""Create embeddings for chunked documents and save as JSONL.

Requires an OpenAI-compatible key in environment variable OPENAI_API_KEY.
If the `openai` package isn't available, the script will error with instructions.

Usage:
  python -m tools.embeddings --in chunks.jsonl --out vectors.jsonl --model text-embedding-3-small
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List

try:
    import openai
except Exception:
    openai = None


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def write_jsonl(items: Iterable[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    if openai is None:
        raise SystemExit("The 'openai' package is required. Install with: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Please set OPENAI_API_KEY in your environment before running this script.")

    openai.api_key = api_key

    # simple batching to avoid too-large requests
    resp = openai.Embedding.create(input=texts, model=model)
    return [r["embedding"] for r in resp["data"]]


def process(in_path: Path, out_path: Path, model: str = "text-embedding-3-small", batch_size: int = 16):
    docs = list(read_jsonl(in_path))
    out_items = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        texts = [d["text"] for d in batch]
        print(f"Embedding batch {i}..{i+len(batch)-1}")
        vectors = embed_texts(texts, model=model)
        for doc, vec in zip(batch, vectors):
            item = {"id": doc["id"], "title": doc.get("title"), "text": doc.get("text"), "embedding": vec}
            out_items.append(item)
        time.sleep(0.2)

    write_jsonl(out_items, out_path)
    print(f"Wrote {len(out_items)} vectors to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inpath", required=True)
    p.add_argument("--out", dest="outpath", required=True)
    p.add_argument("--model", default="text-embedding-3-small")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    in_path = Path(args.inpath)
    out_path = Path(args.outpath)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    process(in_path, out_path, model=args.model, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
