"""tools.rag.rag_sql_generator

RAG-based SQL generator: retrieve relevant chunks and call LLM with strict SQL-only prompt.
"""
from __future__ import annotations

import os
from typing import List, Tuple, Optional, Dict
import re
import numpy as np
import logging

from .embed_rules import cosine_similarity_matrix, embed_chunks

try:
    import openai
except Exception:
    openai = None

logger = logging.getLogger(__name__)


class RAGSQLGenerator:
    def __init__(self, chunks: List[str], embeddings: np.ndarray):
        self.chunks = chunks
        self.embeddings = embeddings

    def retrieve(self, query: str, query_emb: Optional[np.ndarray] = None, top_k: int = 4) -> List[Tuple[int, str, float]]:
        """Retrieve top-K relevant chunks based on cosine similarity."""
        if query_emb is None:
            query_emb = embed_chunks([query])[0:1]
        sims = cosine_similarity_matrix(query_emb, self.embeddings)[0]
        idxs = np.argsort(-sims)[:top_k]
        return [(int(i), self.chunks[int(i)], float(sims[int(i)])) for i in idxs]

    def _build_prompt(self, query: str, retrieved: List[Tuple[int, str, float]]) -> str:
        """Constructs a strict SQL-only prompt for the LLM."""
        rules_text = "\n\n".join([f"[CHUNK {i}]\n{chunk}" for i, chunk, _ in retrieved])
        instruction = (
            "You are an expert SQL generator. Use only the business rules provided below "
            "to write a single, safe SQL query that answers the user's natural language question.\n\n"
        )
        fmt = (
            "REQUIREMENTS:\n"
            "- ONLY output SQL. Do NOT output any explanation, markdown, or prose.\n"
            "- Output exactly one SQL statement beginning with SELECT or WITH.\n"
            "- Use column names exactly as in the table.\n"
            "- Handle empty strings and NULLs via COALESCE/NULLIF.\n"
            "- Replace commas with dots before CAST, and use CAST(... AS NUMERIC).\n"
            "- Allowed functions: REPLACE, COALESCE, NULLIF, CAST, SUM, AVG, COUNT, MAX, MIN, ROUND, DATE_TRUNC, EXTRACT.\n"
            "- No DROP/DELETE/UPDATE/INSERT/ALTER/TRUNCATE or multiple statements.\n"
            "- Avoid division by zero using NULLIF(denominator, 0).\n\n"
        )
        prompt = (
            f"{instruction}"
            f"BUSINESS RULES:\n\n{rules_text}\n\n"
            f"USER QUERY: {query}\n\n"
            f"{fmt}"
            "DO NOT change output format. Only produce SQL. Return only SQL."
        )
        return prompt

    def generate_sql(self, query: str, top_k: int = 4, temperature: float = 0.0) -> Tuple[str, Dict[str, str]]:
        """Generates a strict SQL query from natural language using RAG + LLM."""
        # First, if the new text2sql QueryAgent exists in the repo, prefer it as a
        # drop-in replacement for the old RAG-based generator. This keeps a single
        # integration point while allowing incremental migration.
        try:
            from text2sql.agent import QueryAgent

            # Use the QueryAgent to generate SQL directly. Return a (sql, params)
            # tuple to preserve compatibility with older callers.
            agent = QueryAgent()
            res = agent.generate_and_run(query)
            sql = res.get("sql") or ""
            return sql, {}
        except Exception:
            # If text2sql is not available or fails, fall back to original RAG flow
            pass

        q_emb = embed_chunks([query])[0:1]
        retrieved = self.retrieve(query, query_emb=q_emb, top_k=top_k)
        prompt = self._build_prompt(query, retrieved)

        # Load configuration
        openai_key = os.environ.get("OPENAI_API_KEY")
        groq_key = os.environ.get("GROQ_API_KEY")
        force_openai = os.environ.get("FORCE_OPENAI", "0") == "1"
        prefer_groq = os.environ.get("PREFER_GROQ", "1") == "1"
        model_name = os.environ.get("OPENAI_CHAT_MODEL", "llama-3.3-70b-versatile")
        system_msg = "You are a strict SQL-only assistant. Follow instructions carefully."

        sql = ""

        # Try Groq backend first (preferred)
        if (groq_key or prefer_groq) and not (force_openai and openai_key):
            try:
                logger.debug("RAG using Groq backend (model=%s)", model_name)
                from llm.client import get_llm_response

                resp = get_llm_response(
                    prompt,
                    model=model_name,
                    max_tokens=1024,
                    temperature=temperature,
                    system_prompt=system_msg,
                )
                sql = resp.get("content", "").strip()
            except Exception as e:
                logger.debug("Groq backend failed for RAG: %s", e)
                # Try OpenAI fallback
                if openai is not None and openai_key:
                    logger.debug("Falling back to OpenAI for RAG generation")
                    openai.api_key = openai_key
                    resp = openai.ChatCompletion.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=temperature,
                        max_tokens=1024,
                    )
                    sql = resp["choices"][0]["message"]["content"].strip()
                else:
                    raise RuntimeError(
                        "No LLM backend available for RAG generation or Groq failed. "
                        "Set GROQ_API_KEY or OPENAI_API_KEY."
                    )

        else:
            # OpenAI only
            if openai is None or not openai_key:
                raise RuntimeError("OpenAI SDK not configured for RAG generation")
            logger.debug("RAG using OpenAI backend (model=%s)", model_name)
            openai.api_key = openai_key
            resp = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=1024,
            )
            sql = resp["choices"][0]["message"]["content"].strip()

        # --- Sanity check ---
        def _looks_like_sql(text: str) -> bool:
            if not text:
                return False
            s = text.strip()
            if s.startswith("```"):
                s = "\n".join(s.splitlines()[1:]).strip()
            s_low = s.lstrip().lower()
            return s_low.startswith("select") or s_low.startswith("with")

        if not _looks_like_sql(sql):
            logger.debug("RAG LLM response did not look like SQL: %r", (sql or '')[:200])
            raise RuntimeError("RAG LLM backend returned non-SQL response; see logs for details")

        if sql.startswith("```"):
            lines = sql.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            sql = "\n".join(lines).strip()

        # Parameterize any inline single-quoted string literals to avoid returning
        # SQL with embedded user-provided values. This converts literals like
        #  'En baisse' -> :p1 and returns params={'p1': 'En baisse'}
        params: Dict[str, str] = {}

        def _replace_literal(m: re.Match) -> str:
            # captured content without surrounding quotes; handle doubled '' -> '
            inner = m.group(1).replace("''", "'")
            key = f"p{len(params) + 1}"
            params[key] = inner
            return f":{key}"

        try:
            sql = re.sub(r"'([^']*(?:''[^']*)*)'", _replace_literal, sql)
        except Exception:
            # If something unexpected happens, fallback to returning original SQL
            params = {}

        return sql, params
