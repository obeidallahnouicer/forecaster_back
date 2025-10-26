"""tools.rag package

This package contains RAG utilities for embedding business rules and generating
strict, SQL-only outputs from an LLM. Modules:
- embed_rules: markdown loading, chunking, embedding (local or OpenAI)
- rag_sql_generator: retrieval + prompt builder + LLM call
- rag_sql_validator: lightweight SQL safety checks
"""
from .embed_rules import load_markdown, chunk_text, embed_chunks, build_embeddings_from_file  # noqa: F401
from .rag_sql_generator import RAGSQLGenerator  # noqa: F401
from .rag_sql_validator import is_safe_sql, enforce_sql_format  # noqa: F401
