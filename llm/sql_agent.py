"""Thin adapter that builds a prompt with the SQL prompt builder and calls the LLM client.

This module provides a small helper to convert a user's natural language question
into a prompt (including the first two sample rows) and send it to the LLM via
the project's llm.client.get_llm_response function. It parses JSON responses
and returns a normalized dict with sql, params, explanation, raw, and meta.
"""
from typing import List, Dict, Optional, Any
import json

from prompts import sql_generation as sg
from llm.client import get_llm_response


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
    prompt = sg.build_sql_generation_prompt(question, sample_rows, stock_columns)

    resp = get_llm_response(prompt, model=model, max_tokens=max_tokens, temperature=temperature, system_prompt=sg.SQL_SYSTEM_PROMPT)
    content = resp.get("content", "")

    # Try to parse JSON output from the model
    try:
        parsed = json.loads(content)
        # Normalize result shape
        return {
            "sql": parsed.get("sql", ""),
            "params": parsed.get("params", {}),
            "explanation": parsed.get("explanation", ""),
            "raw": content,
            "meta": resp,
        }
    except Exception:
        # Return helpful debug info when parsing fails
        return {
            "sql": "",
            "params": {},
            "explanation": "LLM did not return valid JSON",
            "raw": content,
            "meta": resp,
        }
