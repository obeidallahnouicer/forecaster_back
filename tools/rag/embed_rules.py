"""tools.rag.embed_rules
Utilities to load business rules markdown files, chunk them, and compute embeddings.

This module prefers `sentence-transformers` for local embeddings and falls back to
OpenAI embeddings if an API key is provided. It exposes simple functions to load
markdown, chunk it, and compute embeddings for a list of text chunks.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _HAS_S2 = True
except Exception:
    _HAS_S2 = False

try:
    import openai
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False


def load_markdown(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    text = text.replace("\r\n", "\n")
    n = len(text)
    chunks: List[str] = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(end - overlap, end)
    return chunks


def _openai_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    if not _HAS_OPENAI:
        raise RuntimeError("openai package not installed")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    openai.api_key = api_key
    resp = openai.Embeddings.create(model=model, input=texts)
    embs = [e["embedding"] for e in resp["data"]]
    return np.array(embs, dtype=float)


def _s2_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    if not _HAS_S2:
        raise RuntimeError("sentence-transformers not available")
    if not hasattr(_s2_embeddings, "_model"):
        _s2_embeddings._model = SentenceTransformer(model_name)
    model = _s2_embeddings._model
    embs = model.encode(texts, show_progress_bar=False)
    return np.array(embs, dtype=float)


def embed_chunks(chunks: List[str], prefer_local: bool = True) -> np.ndarray:
    if prefer_local and _HAS_S2:
        return _s2_embeddings(chunks)
    if _HAS_OPENAI and os.environ.get("OPENAI_API_KEY"):
        return _openai_embeddings(chunks)
    if _HAS_S2:
        return _s2_embeddings(chunks)
    raise RuntimeError(
        "No embedding backend available. Install sentence-transformers or set OPENAI_API_KEY and install openai."
    )


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    def _norm(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return x / norms
    a_n = _norm(a)
    b_n = _norm(b)
    return np.dot(a_n, b_n.T)


def build_embeddings_from_file(path: str, chunk_size: int = 1200, overlap: int = 200) -> Tuple[List[str], np.ndarray]:
    text = load_markdown(path)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    embs = embed_chunks(chunks)
    return chunks, embs
