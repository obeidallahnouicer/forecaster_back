"""
Integration tests for the RAG chatbot module.

Run with:
    pytest tests/test_chatbot_integration.py -v
"""

import pytest
import tempfile
import logging
from pathlib import Path
import pandas as pd
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_chatbot import chatbot, config, indexer, retriever

logging.basicConfig(level=logging.INFO)


@pytest.fixture
def sample_csv(tmp_path):
    """Create a small sample CSV for testing."""
    data = {
        "ref_article": ["PA0001", "PA0002", "PA0003", "PA0004", "PA0005"],
        "designation": [
            "Product A",
            "Product B",
            "Product C",
            "Product D",
            "Product E",
        ],
        "marque": ["Brand1", "Brand2", "Brand1", "Brand3", "Brand2"],
        "famille": ["Family1", "Family1", "Family2", "Family2", "Family1"],
        "next_year": [1000, 2000, 1500, 800, 2500],
        "avg_forecast": [950, 2100, 1400, 850, 2400],
        "trend_label": ["Uptrend", "Stable", "Downtrend", "Uptrend", "Stable"],
    }
    
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    
    return csv_path, df


@pytest.fixture
def vectorstore_dir(tmp_path):
    """Create a temporary directory for vectorstore."""
    return tmp_path / "vectorstore"


@pytest.mark.slow
class TestChatbotIntegration:
    """Integration tests for chatbot with vectorstore."""
    
    def test_get_answer_returns_structured_response(self, sample_csv, vectorstore_dir):
        """Test that get_answer returns properly structured response."""
        csv_path, df = sample_csv
        
        # Load data
        df = indexer.load_and_normalize_csv(csv_path)
        
        # Build vectorstore
        chunks = indexer.chunk_texts(df, chunk_size=500)
        embeddings = indexer.compute_embeddings_in_batches(
            chunks,
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=32,
            device="cpu",
        )
        store = indexer.upsert_to_chroma(
            vectorstore_dir,
            chunks,
            embeddings,
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            force_recreate=True,
        )
        
        # Mock the retriever
        import unittest.mock as mock
        with mock.patch("rag_chatbot.chatbot.get_vectorstore", return_value=store):
            result = chatbot.get_answer("top products", top_k=3)
        
        # Verify structure
        assert "answer" in result
        assert "llm_prompt" in result
        assert "sources" in result
        assert "retrieved_docs" in result
        assert "embedding_model" in result
        assert "num_chunks" in result
        
        # Verify content
        assert result["answer"] is None  # Should be None; user provides to LLM
        assert isinstance(result["llm_prompt"], str)
        assert len(result["llm_prompt"]) > 0
        assert len(result["sources"]) > 0
        assert len(result["retrieved_docs"]) > 0
    
    def test_retrieved_docs_have_required_fields(self, sample_csv, vectorstore_dir):
        """Test that retrieved docs have id, content, and metadata."""
        csv_path, df = sample_csv
        
        df = indexer.load_and_normalize_csv(csv_path)
        chunks = indexer.chunk_texts(df, chunk_size=500)
        embeddings = indexer.compute_embeddings_in_batches(
            chunks,
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=32,
            device="cpu",
        )
        store = indexer.upsert_to_chroma(
            vectorstore_dir,
            chunks,
            embeddings,
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            force_recreate=True,
        )
        
        import unittest.mock as mock
        with mock.patch("rag_chatbot.chatbot.get_vectorstore", return_value=store):
            result = chatbot.get_answer("product", top_k=2)
        
        for doc in result["retrieved_docs"]:
            assert "id" in doc
            assert "content" in doc
            assert "full_content" in doc
            assert "metadata" in doc
            assert len(doc["content"]) > 0
            assert len(doc["full_content"]) > 0
    
    def test_chat_stateful_memory(self, sample_csv, vectorstore_dir):
        """Test that chat maintains stateful memory across calls."""
        csv_path, df = sample_csv
        thread_id = "test_thread_123"
        
        # Clean up any existing memory
        chatbot.reset_memory(thread_id)
        
        # Build vectorstore
        df = indexer.load_and_normalize_csv(csv_path)
        chunks = indexer.chunk_texts(df, chunk_size=500)
        embeddings = indexer.compute_embeddings_in_batches(
            chunks,
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=32,
            device="cpu",
        )
        store = indexer.upsert_to_chroma(
            vectorstore_dir,
            chunks,
            embeddings,
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            force_recreate=True,
        )
        
        import unittest.mock as mock
        with mock.patch("rag_chatbot.chatbot.get_vectorstore", return_value=store):
            # First message
            result1 = chatbot.chat("What is Product A?", thread_id)
            assert result1["thread_id"] == thread_id
            assert result1["answer"] is not None
            
            # Check memory
            mem = chatbot.get_memory(thread_id)
            assert len(mem.chat_memory.messages) >= 2  # At least user + assistant
            
            # Second message
            result2 = chatbot.chat("What about Product B?", thread_id)
            assert result2["thread_id"] == thread_id
            
            # Memory should grow
            mem = chatbot.get_memory(thread_id)
            assert len(mem.chat_memory.messages) >= 4  # 2 more messages
        
        # Clean up
        chatbot.reset_memory(thread_id)
    
    def test_sources_match_retrieved_docs(self, sample_csv, vectorstore_dir):
        """Test that sources list matches retrieved doc IDs."""
        csv_path, df = sample_csv
        
        df = indexer.load_and_normalize_csv(csv_path)
        chunks = indexer.chunk_texts(df, chunk_size=500)
        embeddings = indexer.compute_embeddings_in_batches(
            chunks,
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=32,
            device="cpu",
        )
        store = indexer.upsert_to_chroma(
            vectorstore_dir,
            chunks,
            embeddings,
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            force_recreate=True,
        )
        
        import unittest.mock as mock
        with mock.patch("rag_chatbot.chatbot.get_vectorstore", return_value=store):
            result = chatbot.get_answer("test query", top_k=3)
        
        # Sources should match doc IDs
        source_ids = [doc["id"] for doc in result["retrieved_docs"]]
        assert result["sources"] == source_ids
    
    def test_embedding_model_from_metadata(self, sample_csv, vectorstore_dir):
        """Test that embedding model is correctly read from metadata."""
        csv_path, df = sample_csv
        
        df = indexer.load_and_normalize_csv(csv_path)
        chunks = indexer.chunk_texts(df, chunk_size=500)
        embeddings = indexer.compute_embeddings_in_batches(
            chunks,
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=32,
            device="cpu",
        )
        store = indexer.upsert_to_chroma(
            vectorstore_dir,
            chunks,
            embeddings,
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            force_recreate=True,
        )
        
        # Save metadata
        indexer.save_vectorstore_metadata(
            vectorstore_dir,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            chunk_size=500,
            chunk_overlap=50,
            batch_size=32,
            num_docs=len(chunks),
            csv_hash="abc123",
        )
        
        import unittest.mock as mock
        with mock.patch("rag_chatbot.chatbot.get_vectorstore", return_value=store):
            result = chatbot.get_answer("query", top_k=2)
        
        assert result["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert result["num_chunks"] == len(chunks)


class TestMemoryPersistence:
    """Test conversation memory persistence."""
    
    def test_memory_persisted_to_disk(self, tmp_path):
        """Test that memory is saved to disk."""
        thread_id = f"persist_test_{id(tmp_path)}"
        
        # Create a memory and add messages
        mem = chatbot.get_memory(thread_id)
        mem.chat_memory.add_user_message("test message 1")
        mem.chat_memory.add_ai_message("test response 1")
        
        # Persist it
        chatbot._persist_memory(thread_id)
        
        # Check file exists
        mem_file = Path(config.MEMORY_DIR) / f"{thread_id}.json"
        assert mem_file.exists()
        
        # Load and verify
        data = json.loads(mem_file.read_text(encoding="utf-8"))
        assert len(data) == 2
        assert data[0]["role"] == "user"
        assert data[0]["text"] == "test message 1"
        assert data[1]["role"] == "ai"
        
        # Clean up
        chatbot.reset_memory(thread_id)
        assert not mem_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
