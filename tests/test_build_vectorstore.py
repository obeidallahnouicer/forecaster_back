"""
Test suite for RAG vectorstore indexing pipeline.

Run with:
    pytest tests/test_build_vectorstore.py -v
    
Or in the workspace:
    python -m pytest tests/test_build_vectorstore.py -q
"""

import pytest
import tempfile
import shutil
import logging
from pathlib import Path
import pandas as pd
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_chatbot import indexer, config

# Setup logging for tests
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


class TestLoadAndNormalize:
    """Test CSV loading and normalization."""
    
    def test_load_csv(self, sample_csv):
        """Test loading CSV file."""
        csv_path, original_df = sample_csv
        
        df = indexer.load_and_normalize_csv(csv_path)
        
        assert len(df) == 5
        assert len(df.columns) == 7
        assert "ref_article" in df.columns
    
    def test_normalize_whitespace(self, sample_csv):
        """Test that whitespace is normalized."""
        csv_path, original_df = sample_csv
        
        df = indexer.load_and_normalize_csv(csv_path)
        
        # Check that values are cleaned
        for col in df.select_dtypes(include='object').columns:
            for val in df[col]:
                if pd.notna(val):
                    # No leading/trailing whitespace
                    assert str(val) == str(val).strip()


class TestChunking:
    """Test text chunking logic."""
    
    def test_chunk_creation(self, sample_csv):
        """Test that chunks are created."""
        csv_path, df = sample_csv
        
        chunks = indexer.chunk_texts(
            df,
            chunk_size=500,
            chunk_overlap=50,
            max_rows=None,
        )
        
        assert len(chunks) > 0
        
        # Each chunk should have required fields
        for chunk in chunks:
            assert "id" in chunk
            assert "page_content" in chunk
            assert "metadata" in chunk
            assert chunk["page_content"]  # Non-empty content
    
    def test_deterministic_ids(self, sample_csv):
        """Test that chunk IDs are deterministic."""
        csv_path, df = sample_csv
        
        chunks1 = indexer.chunk_texts(df, chunk_size=500, chunk_overlap=50)
        chunks2 = indexer.chunk_texts(df, chunk_size=500, chunk_overlap=50)
        
        # Same data should produce same IDs
        ids1 = [c["id"] for c in chunks1]
        ids2 = [c["id"] for c in chunks2]
        
        assert ids1 == ids2
    
    def test_max_rows(self, sample_csv):
        """Test max_rows parameter."""
        csv_path, df = sample_csv
        
        chunks = indexer.chunk_texts(
            df,
            chunk_size=500,
            chunk_overlap=50,
            max_rows=2,
        )
        
        # Should only process 2 rows
        row_indices = set(c["metadata"]["source_row"] for c in chunks)
        assert len(row_indices) <= 2


class TestEmbedding:
    """Test embedding computation (mock or skip on CPU-only if slow)."""
    
    @pytest.mark.slow
    def test_embeddings_shape(self, sample_csv):
        """Test that embeddings have correct shape."""
        csv_path, df = sample_csv
        
        # Use small dev model for speed
        chunks = indexer.chunk_texts(df, chunk_size=500)
        
        embeddings_dict = indexer.compute_embeddings_in_batches(
            chunks,
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=32,
            device="cpu",
        )
        
        assert len(embeddings_dict) == len(chunks)
        
        # Check shapes
        for doc_id, emb in embeddings_dict.items():
            assert isinstance(emb, __import__("numpy").ndarray)
            assert emb.ndim == 1
            assert len(emb) > 0  # Non-empty


class TestConfigHash:
    """Test configuration hashing for cache invalidation."""
    
    def test_config_hash_deterministic(self):
        """Test that config hash is deterministic."""
        hash1 = indexer.compute_config_hash(
            embedding_model="test-model",
            chunk_size=2048,
            chunk_overlap=128,
            batch_size=64,
        )
        
        hash2 = indexer.compute_config_hash(
            embedding_model="test-model",
            chunk_size=2048,
            chunk_overlap=128,
            batch_size=64,
        )
        
        assert hash1 == hash2
    
    def test_config_hash_differs(self):
        """Test that different configs produce different hashes."""
        hash1 = indexer.compute_config_hash(
            embedding_model="model-a",
            chunk_size=2048,
            chunk_overlap=128,
            batch_size=64,
        )
        
        hash2 = indexer.compute_config_hash(
            embedding_model="model-b",
            chunk_size=2048,
            chunk_overlap=128,
            batch_size=64,
        )
        
        assert hash1 != hash2


class TestCSVHash:
    """Test CSV content hashing."""
    
    def test_csv_hash_deterministic(self, sample_csv):
        """Test that CSV hash is deterministic."""
        csv_path, df = sample_csv
        
        hash1 = indexer.compute_csv_hash(df)
        hash2 = indexer.compute_csv_hash(df)
        
        assert hash1 == hash2
    
    def test_csv_hash_differs_with_rows(self, sample_csv):
        """Test that hash differs when rows change."""
        csv_path, df = sample_csv
        
        hash1 = indexer.compute_csv_hash(df)
        
        # Remove a row
        df_modified = df.iloc[:-1]
        hash2 = indexer.compute_csv_hash(df_modified)
        
        assert hash1 != hash2


class TestMetadata:
    """Test metadata saving and loading."""
    
    def test_save_and_load_metadata(self, vectorstore_dir):
        """Test metadata persistence."""
        vectorstore_dir.mkdir(exist_ok=True)
        
        indexer.save_vectorstore_metadata(
            vectorstore_dir,
            embedding_model="test-model",
            chunk_size=2048,
            chunk_overlap=128,
            batch_size=64,
            num_docs=100,
            csv_hash="abc123",
        )
        
        # Load it back
        meta = indexer.load_vectorstore_metadata(vectorstore_dir)
        
        assert meta is not None
        assert meta["embedding_model"] == "test-model"
        assert meta["chunk_size"] == 2048
        assert meta["num_docs"] == 100
    
    def test_load_missing_metadata(self, vectorstore_dir):
        """Test loading from non-existent directory."""
        meta = indexer.load_vectorstore_metadata(vectorstore_dir)
        assert meta is None


class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.slow
    def test_build_vectorstore_e2e(self, sample_csv, vectorstore_dir):
        """Test complete pipeline: load -> chunk -> embed -> upsert."""
        csv_path, df = sample_csv
        
        # Load data
        df = indexer.load_and_normalize_csv(csv_path)
        
        # Chunk
        chunks = indexer.chunk_texts(df, chunk_size=500, chunk_overlap=50, max_rows=None)
        assert len(chunks) > 0, "Should create chunks"
        
        # Embed (using small model)
        embeddings_dict = indexer.compute_embeddings_in_batches(
            chunks,
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=32,
            device="cpu",
        )
        assert len(embeddings_dict) == len(chunks), "Should embed all chunks"
        
        # Upsert to Chroma
        store = indexer.upsert_to_chroma(
            vectorstore_dir,
            chunks,
            embeddings_dict,
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            force_recreate=True,
        )
        
        assert store is not None, "Should return Chroma store"
        
        # Try a query
        results = store.similarity_search("Product A", k=1)
        assert len(results) > 0, "Should find results"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
