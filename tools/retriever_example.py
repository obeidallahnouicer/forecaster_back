"""Simple file-based retriever using stored embeddings (vectors.jsonl).

This is a minimal offline retriever that computes cosine similarity between a query
and stored embeddings. It uses OpenAI embeddings for the query (same model) by default.

Usage:
  python -m tools.retriever_example --vectors vectors.jsonl --query "Which clients are inactive for 3 months?" --topk 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Tuple

try:
    import openai
except Exception:
    openai = None

import numpy as np


def read_vectors(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embed_query(query: str, model: str = "text-embedding-3-small") -> List[float]:
    if openai is None:
        raise SystemExit("openai package required for query embedding. pip install openai")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Please set OPENAI_API_KEY in your environment.")
    openai.api_key = api_key
    resp = openai.Embedding.create(input=[query], model=model)
    return resp["data"][0]["embedding"]


def retrieve(vectors_path: Path, query: str, topk: int = 4) -> List[Tuple[float, dict]]:
    items = read_vectors(vectors_path)
    qvec = np.array(embed_query(query))
    scored = []
    for it in items:
        emb = np.array(it["embedding"])
        score = cosine_sim(qvec, emb)
        scored.append((score, it))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:topk]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vectors", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--topk", type=int, default=4)
    args = p.parse_args()

    results = retrieve(Path(args.vectors), args.query, topk=args.topk)
    for score, it in results:
        print(f"SCORE: {score:.4f}  TITLE: {it.get('title')}")
        print(it.get("text")[:1000])
        print("-" * 80)


if __name__ == "__main__":
    main()
