"""Thin adapter that builds a prompt with the SQL prompt builder and calls the LLM client.

This module provides a small helper to convert a user's natural language question
into a prompt (including the first two sample rows) and send it to the LLM via
the project's llm.client.get_llm_response function.

New behavior:
 - The LLM is instructed to return a JSON object containing at minimum the keys
     `reasoning` (short explanation) and `sql_query` (the SELECT statement string).
 - This function is backwards-compatible: if the model returns the older keys
     (`sql`, `params`, `explanation`) they will be mapped to the new shape.
 - The returned dict is validated to ensure `sql_query` is a SELECT statement
     (basic validation). If validation fails, the function returns a helpful
     explanation and preserves the raw LLM output in `raw`/`meta` for debugging.
"""
from typing import List, Dict, Optional, Any
import json
import os
import numpy as np

from prompts import sql_generation as sg
from llm.client import get_llm_response
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

# Optional RAG-based generator integration
try:
    from tools.rag.rag_sql_generator import RAGSQLGenerator
    from tools.rag.embed_rules import build_embeddings_from_file
    _HAS_RAG = True
except Exception:
    RAGSQLGenerator = None  # type: ignore
    build_embeddings_from_file = None  # type: ignore
    _HAS_RAG = False

# Allow runtime toggle for RAG (set ENABLE_RAG=0 to disable)
_RAG_ENABLED = _HAS_RAG and os.getenv("ENABLE_RAG", "1") != "0"


def _find_rules_file() -> str:
    """Look for TABLE Chatbot markdown file in repo root.

    Returns path or raises FileNotFoundError.
    """
    candidates = ["TABLE_Chatbot.md", "TABLE Chatbot.md", "TABLE_Chatbot.MD", "TABLE Chatbot.MD"]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("Business rules markdown not found: looked for TABLE_Chatbot.md or TABLE Chatbot.md")


def _load_or_build_embeddings(cache_path: str = "cache/rag_embeddings.npz"):
    """Load cached chunks+embeddings or build them from the markdown file.

    Returns (chunks: List[str], embeddings: np.ndarray)
    """
    if not _HAS_RAG:
        raise RuntimeError("RAG support not available (missing modules)")

    # Try cache first
    try:
        if os.path.exists(cache_path):
            npz = np.load(cache_path, allow_pickle=True)
            chunks = list(npz["chunks"].tolist())
            embeddings = npz["embeddings"]
            return chunks, embeddings
    except Exception:
        # fall through to rebuild
        pass

    # Build from markdown
    rules_path = _find_rules_file()
    chunks, embeddings = build_embeddings_from_file(rules_path)

    # Save cache (best-effort) - ensure directory exists
    try:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache_path, chunks=np.array(chunks, dtype=object), embeddings=embeddings)
    except Exception as e:
        logger.debug("Failed to write RAG cache (%s): %s", cache_path, e)
        # ignore cache failures
        pass

    return chunks, embeddings


def generate_sql_from_question(
    question: str,
    sample_rows: List[Dict[str, Any]],
    stock_columns: Optional[List[str]] = None,
    model: str = "llama-3.3-70b-versatile",
    max_tokens: int = 500,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Build prompt, call LLM, and parse JSON response.

    Returns a dict with keys:
      - sql: the SELECT SQL string (or empty string)
      - params: dict of parameters (or empty dict)
      - explanation: brief explanation string
      - raw: raw LLM content
      - meta: the raw get_llm_response return value for debugging
    """
    # Try to use RAG-based generator if available and enabled and business rules file present.
    if _RAG_ENABLED:
        try:
            chunks, embeddings = _load_or_build_embeddings()
            rag = RAGSQLGenerator(chunks, embeddings)
            rag_out = rag.generate_sql(question, top_k=4, temperature=0.0)
            # rag.generate_sql may return either a plain SQL string (legacy) or
            # a tuple (sql, params). Handle both shapes for backwards compat.
            if isinstance(rag_out, tuple) and len(rag_out) == 2:
                sql_text, params = rag_out
            else:
                sql_text, params = (rag_out or "", {})

            return {
                "reasoning": "RAG-generated SQL using business rules",
                "sql_query": sql_text or "",
                "params": params or {},
                "raw": sql_text or "",
                "meta": {"method": "rag"},
            }
        except FileNotFoundError:
            # no rules file; fall back to prompt-based approach
            logger.debug("RAG rules file not found; falling back to prompt-based generation.")
        except Exception as e:
            # If RAG generation fails for any reason, fall back quietly to existing flow
            logger.exception("RAG generation failed, falling back to prompt-based: %s", e)
            pass
    else:
        if _HAS_RAG:
            logger.debug("RAG available but disabled via ENABLE_RAG=0")
        else:
            logger.debug("RAG not available in this environment")

    # Fallback: original prompt-based generation
    prompt = sg.build_sql_generation_prompt(question, sample_rows, stock_columns)

    resp = get_llm_response(
        prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=sg.SQL_SYSTEM_PROMPT,
    )
    content = resp.get("content", "")

    # Delegate parsing to helper so it can be unit-tested separately.
    return parse_and_normalize_llm_response(content, resp)


def parse_and_normalize_llm_response(content: str, resp: Dict[str, Any]) -> Dict[str, Any]:
    """Parse model content (string) and normalize to new output shape.

    Returns dict with keys: reasoning, sql_query, params, raw, meta.
    This function is safe to call in unit tests with a fake `resp` dict.
    """
    # Try to parse JSON output from the model and normalize to the new shape:
    # { reasoning: str, sql_query: str, params: dict }
    try:
        parsed = json.loads(content)
    except Exception:
        return {
            "reasoning": "LLM did not return valid JSON",
            "sql_query": "",
            "params": {},
            "raw": content,
            "meta": resp,
        }

    # Backwards compatibility: map older keys to new names
    reasoning = None
    sql_query = None
    params = {}

    if isinstance(parsed, dict):
        # Preferred new names
        reasoning = parsed.get("reasoning")
        sql_query = parsed.get("sql_query") or parsed.get("sql")
        params = parsed.get("params") or {}
        # Older responses used 'explanation' instead of 'reasoning'
        if not reasoning:
            reasoning = parsed.get("explanation")

    # Ensure types
    if not isinstance(reasoning, (str, type(None))):
        reasoning = str(reasoning)
    if not isinstance(sql_query, (str, type(None))):
        sql_query = str(sql_query)
    if not isinstance(params, dict):
        params = {}

    # Basic validation: sql_query should start with SELECT (case-insensitive)
    if sql_query:
        if not sql_query.strip().lower().startswith("select"):
            return {
                "reasoning": (
                    "Model returned SQL that does not start with SELECT. "
                    "Refusing to return non-SELECT statements."
                ),
                "sql_query": "",
                "params": {},
                "raw": content,
                "meta": resp,
            }

    # Final normalized result
    return {
        "reasoning": reasoning or "",
        "sql_query": sql_query or "",
        "params": params or {},
        "raw": content,
        "meta": resp,
    }
