"""Backward-compatible shim to `tools.rag.rag_sql_generator`.

This file keeps code that expects `rag_sql_generator` at the repo root working.
It re-exports the primary `RAGSQLGenerator` symbol from the package version.
"""
from tools.rag.rag_sql_generator import *  # noqa: F401,F403
