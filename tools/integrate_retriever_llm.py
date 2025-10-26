"""Integration script: retrieve top chunks and call the project's LLM client.

Usage:
  python -m tools.integrate_retriever_llm --vectors vectors.jsonl --question "Why is stock low for ARGAN 500ML?" --topk 3
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from tools.retriever_example import retrieve
from llm.client import get_llm_response


SYSTEM_PROMPT = (
    "You are a domain-aware SQL and insights generator. Use ONLY the provided context. "
    "Follow the business rules and thresholds present in the context. Return a short answer and, if requested, a SQL query. "
    "When returning SQL, include only the SQL inside a single fenced code block and then a one-paragraph justification."
)


def build_prompt(chunks: List[dict], question: str) -> str:
    parts = ["Context (most relevant chunks):\n"]
    for i, c in enumerate(chunks):
        parts.append(f"--- Chunk {i+1}: {c.get('title', '')}\n{c.get('text','')}\n")

    parts.append("\nUser question:\n" + question + "\n")
    parts.append("\nInstructions:\n- Use the context above.");
    parts.append("- If generating SQL, reference only tables/columns mentioned in the context.")
    return "\n".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vectors", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--topk", type=int, default=3)
    args = p.parse_args()

    vec_path = Path(args.vectors)
    if not vec_path.exists():
        raise SystemExit(f"Vectors file not found: {vec_path}")

    # Retrieve top chunks (this will embed the query using OpenAI by default)
    scored = retrieve(vec_path, args.question, topk=args.topk)
    chunks = [it for score, it in scored]

    prompt = build_prompt(chunks, args.question)

    # Call project's LLM client
    resp = get_llm_response(prompt, system_prompt=SYSTEM_PROMPT, max_tokens=800, temperature=0.0)

    print("--- LLM RESPONSE ---")
    print(resp.get("content"))
    print("\n--- usage ---")
    print(f"Model: {resp.get('model')}  prompt_tokens: {resp.get('prompt_tokens')} completion_tokens: {resp.get('completion_tokens')}")


if __name__ == "__main__":
    main()
