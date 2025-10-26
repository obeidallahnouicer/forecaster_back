import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import llm.sql_agent as sql_agent


class DummyRAG:
    def __init__(self, chunks, embeddings):
        self.chunks = chunks
        self.embeddings = embeddings

    def generate_sql(self, query, top_k=4, temperature=0.0):
        # return a deterministic SQL for testing
        return "SELECT id, name FROM t_users ORDER BY id DESC LIMIT 5"


def test_rag_path_monkeypatch(monkeypatch, tmp_path):
    # Monkeypatch RAG availability and loader
    monkeypatch.setattr(sql_agent, '_RAG_ENABLED', True)
    monkeypatch.setattr(sql_agent, '_HAS_RAG', True)

    def fake_load(cache_path="cache/rag_embeddings.npz"):
        chunks = ["chunk1", "chunk2"]
        embeddings = np.zeros((2, 8))
        return chunks, embeddings

    monkeypatch.setattr(sql_agent, '_load_or_build_embeddings', fake_load)

    # Monkeypatch RAG generator class
    monkeypatch.setattr('tools.rag.rag_sql_generator.RAGSQLGenerator', DummyRAG)

    res = sql_agent.generate_sql_from_question("list users", sample_rows=[])
    assert res.get('meta', {}).get('method') == 'rag'
    assert 'sql_query' in res
    assert res['sql_query'].lower().startswith('select')
import numpy as np
import pytest

import llm.sql_agent as sql_agent


def test_rag_path_returns_sql(monkeypatch):
    """Test that when RAG is available, the llm.sql_agent returns the RAG SQL."""

    # Fake embeddings and chunks
    fake_chunks = ["rule about clients", "rule about products"]
    fake_embs = np.zeros((2, 3), dtype=float)

    # Monkeypatch the loader so it doesn't require a real TABLE Chatbot.md file
    monkeypatch.setattr(sql_agent, "_load_or_build_embeddings", lambda cache_path="cache/rag_embeddings.npz": (fake_chunks, fake_embs))

    # Monkeypatch the RAG generator to return a known SQL
    expected_sql = "SELECT client_id, client_name FROM clients WHERE months_inactive >= 6"

    class DummyRAG:
        def __init__(self, chunks, embeddings):
            pass

        def generate_sql(self, query, top_k=4, temperature=0.0):
            return expected_sql

    monkeypatch.setattr("tools.rag.rag_sql_generator.RAGSQLGenerator", DummyRAG)

    out = sql_agent.generate_sql_from_question("Quels clients sont inactifs depuis 6 mois ?", sample_rows=[])

    assert out["sql_query"] == expected_sql
    assert out["meta"].get("method") == "rag"


def test_rag_validator_blocks_drop():
    from tools.rag.rag_sql_validator import is_safe_sql

    assert not is_safe_sql("DROP TABLE users")
    assert is_safe_sql("SELECT * FROM clients")
