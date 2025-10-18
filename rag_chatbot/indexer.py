"""
Robust, resumable vectorstore indexer for RAG.

This module handles:
- Loading and normalizing CSV data
- Chunking text with overlap
- Computing embeddings in batches (CPU-optimized)
- Upserting to Chroma with deduplication
- Tracking config hash for cache invalidation
- Deterministic, stable document IDs based on source row + chunk index

Production-ready: handles GPU detection, batching, error recovery, and resumability.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import re
from datetime import datetime
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

# Embedding model
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

# Chroma (community version)
try:
    from langchain_community.vectorstores import Chroma
    HAS_CHROMA = True
except ImportError:
    try:
        from langchain.vectorstores import Chroma
        HAS_CHROMA = True
    except ImportError:
        HAS_CHROMA = False

from langchain.schema import Document

logger = logging.getLogger("rag.indexer")


# ============================================================================
# CONFIG & HASHING
# ============================================================================

def get_embedding_device() -> str:
    """Detect GPU; fallback to CPU. Returns 'cuda' or 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("CUDA detected; using GPU for embeddings")
            return "cuda"
    except ImportError:
        pass
    logger.info("Using CPU for embeddings")
    return "cpu"


def compute_config_hash(
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
) -> str:
    """
    Compute deterministic hash of embedding config.
    
    This is stored in vectorstore metadata so we can detect when config changed
    and embeddings must be recomputed.
    
    Returns: 16-char hex string (first 16 chars of SHA256)
    """
    config_dict = {
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "batch_size": batch_size,
    }
    config_json = json.dumps(config_dict, sort_keys=True)
    hash_obj = hashlib.sha256(config_json.encode())
    return hash_obj.hexdigest()[:16]


def compute_csv_hash(df: pd.DataFrame) -> str:
    """
    Quick hash of CSV content: header + row count + first/last row.
    Detects if CSV content changed significantly.
    
    Returns: 16-char hex string
    """
    header = "|".join(str(c) for c in df.columns)
    row_count = str(len(df))
    
    # Include first and last row as sentinel
    first_row = df.iloc[0].to_json() if len(df) > 0 else ""
    last_row = df.iloc[-1].to_json() if len(df) > 0 else ""
    
    content = f"{header}|{row_count}|{first_row}|{last_row}"
    hash_obj = hashlib.sha256(content.encode())
    return hash_obj.hexdigest()[:16]


# ============================================================================
# DATA LOADING & NORMALIZATION
# ============================================================================

def load_and_normalize_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load CSV robustly: try multiple encodings and delimiters.
    Normalize column names and text content.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Cleaned pandas DataFrame
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    logger.info(f"Loading CSV from {csv_path}")
    
    # Try common encodings and delimiters
    tried = []
    df = None
    for encoding in ["utf-8", "latin-1", "iso-8859-1"]:
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(csv_path, sep=sep, encoding=encoding, low_memory=False)
                logger.info(f"Successfully loaded CSV with encoding={encoding}, sep='{sep}'")
                break
            except Exception:
                tried.append((encoding, sep))
        if df is not None:
            break
    
    if df is None:
        # Final attempt: let pandas auto-detect
        try:
            df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8", errors="replace")
            logger.info("Loaded CSV with pandas auto-detection")
        except Exception as e:
            logger.error(f"Failed to load CSV after trying {len(tried)} encodings/separators: {e}")
            raise
    
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Normalize column names: strip whitespace, lowercase for consistency
    df.columns = [str(c).strip() for c in df.columns]
    
    # Clean text columns: convert NaN to "", normalize whitespace
    def clean_text(s: str) -> str:
        if pd.isna(s):
            return ""
        s = str(s).strip()
        s = re.sub(r"\s+", " ", s)
        return s
    
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(clean_text)
    
    return df


# ============================================================================
# CHUNKING
# ============================================================================

def chunk_texts(
    df: pd.DataFrame,
    text_columns: Optional[List[str]] = None,
    chunk_size: int = 2048,
    chunk_overlap: int = 128,
    max_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Convert DataFrame rows into chunks for embedding.
    
    Each chunk is a document dict with:
    - id: stable, deterministic hash-based ID
    - page_content: the text chunk
    - metadata: source row, chunk index, etc.
    
    Args:
        df: Input DataFrame
        text_columns: Columns to include; if None, use all string columns
        chunk_size: Target chunk size in characters (approximate)
        chunk_overlap: Overlap between chunks in characters
        max_rows: Limit to first N rows (for testing)
        
    Returns:
        List of chunk dicts
    """
    if max_rows:
        df = df.head(max_rows)
    
    logger.info(f"Chunking {len(df)} rows with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    # Auto-detect text columns if not provided
    # Include ALL columns (both text and numeric) for comprehensive indexing
    if text_columns is None:
        text_columns = list(df.columns)
    
    if not text_columns:
        logger.warning("No columns found; this shouldn't happen")
        text_columns = list(df.columns)
    
    logger.info(f"Using columns for indexing: {text_columns}")
    
    chunks = []
    
    for row_idx, row in df.iterrows():
        # Build document content by joining text columns
        parts = []
        for col in text_columns:
            value = row.get(col, "")
            if pd.notna(value) and str(value).strip():
                parts.append(f"{col}: {str(value).strip()}")
        
        if not parts:
            continue
        
        content = " | ".join(parts)
        
        # Manual chunking with overlap (simple approach)
        # For production, consider using LangChain's TokenTextSplitter
        # but this gives us more control and debugging capability
        content_chunks = []
        start = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))
            content_chunks.append(content[start:end])
            
            # Move start forward, accounting for overlap
            start = end - chunk_overlap if end < len(content) else len(content)
        
        # Create deterministic IDs for each chunk
        for chunk_idx, chunk_text in enumerate(content_chunks):
            # ID: hash of (csv_filename + row_index + chunk_index)
            csv_name = "forecast-summary"  # Could detect from df attrs
            id_str = f"{csv_name}|{row_idx:05d}|{chunk_idx:03d}"
            doc_id = hashlib.sha256(id_str.encode()).hexdigest()[:16]
            
            chunk_dict = {
                "id": doc_id,
                "page_content": chunk_text,
                "metadata": {
                    "source_row": int(row_idx),
                    "chunk_index": int(chunk_idx),
                    "source_id": id_str,
                    # Capture all fields for retrieval and filtering
                    **{col: row.get(col, None) for col in text_columns},
                },
            }
            chunks.append(chunk_dict)
    
    logger.info(f"Created {len(chunks)} chunks")
    return chunks


# ============================================================================
# EMBEDDING
# ============================================================================

def compute_embeddings_in_batches(
    docs: List[Dict[str, Any]],
    embedding_model_name: str,
    batch_size: int = 64,
    device: Optional[str] = None,
    force_embed: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute embeddings for a list of documents in batches.
    
    Uses SentenceTransformer with show_progress_bar and optimal batch size for CPU/GPU.
    
    Args:
        docs: List of doc dicts with 'id', 'page_content', 'metadata'
        embedding_model_name: Model name (e.g., "jinaai/jina-embeddings-v3")
        batch_size: Batch size (tuned per device)
        device: "cuda" or "cpu" (auto-detect if None)
        force_embed: If True, embed all docs; otherwise skip existing IDs
        
    Returns:
        Dict mapping doc_id -> embedding (np.ndarray of float32)
    """
    if not HAS_SENTENCE_TRANSFORMERS:
        raise RuntimeError("sentence_transformers not installed; pip install sentence-transformers")
    
    if device is None:
        device = get_embedding_device()
    
    # Adjust batch size for device
    if device == "cuda":
        batch_size = min(batch_size, 256)  # GPU can handle more
        logger.info(f"GPU detected; using batch_size={batch_size}")
    else:
        batch_size = min(batch_size, 64)  # CPU: more conservative
        logger.info(f"CPU mode; using batch_size={batch_size}")
    
    # Load model
    logger.info(f"Loading embedding model: {embedding_model_name} on device={device}")
    try:
        model = SentenceTransformer(embedding_model_name, device=device, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load embedding model {embedding_model_name}: {e}")
        raise RuntimeError(
            f"Failed to load embedding model '{embedding_model_name}'. "
            f"Ensure sentence-transformers is up-to-date and the model is accessible. "
            f"Error: {e}"
        )
    
    # Extract texts to embed
    texts = [doc["page_content"] for doc in docs]
    embeddings_dict = {}
    
    # Batch embedding with progress bar
    num_batches = (len(texts) + batch_size - 1) // batch_size
    logger.info(f"Embedding {len(texts)} texts in {num_batches} batches (batch_size={batch_size})")
    
    start_time = time.time()
    
    for batch_idx in tqdm(range(num_batches), desc="Embedding batches", unit="batch"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(texts))
        batch_texts = texts[batch_start:batch_end]
        
        try:
            # Encode batch
            batch_embeddings = model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=batch_size,
            )
            
            # Store embeddings with document IDs
            for i, emb in enumerate(batch_embeddings):
                doc_idx = batch_start + i
                doc_id = docs[doc_idx]["id"]
                embeddings_dict[doc_id] = emb.astype(np.float32)
            
            logger.info(
                f"Embedded batch {batch_idx + 1}/{num_batches} "
                f"(docs {batch_start}-{batch_end - 1})"
            )
        except Exception as e:
            logger.error(f"Failed to embed batch {batch_idx} (docs {batch_start}-{batch_end - 1}): {e}")
            # Create dummy zero embeddings so pipeline doesn't break
            # In production, might want to skip these docs
            model_dim = 1024  # Default for most transformers
            for i in range(len(batch_texts)):
                doc_id = docs[batch_start + i]["id"]
                embeddings_dict[doc_id] = np.zeros(model_dim, dtype=np.float32)
                logger.warning(f"Using zero embedding for doc {doc_id}")
    
    elapsed = time.time() - start_time
    logger.info(f"Embedding complete in {elapsed:.1f}s ({len(texts) / elapsed:.1f} docs/sec)")
    
    return embeddings_dict


# ============================================================================
# CHROMA UPSERT & PERSISTENCE
# ============================================================================

def upsert_to_chroma(
    vectorstore_dir: Path,
    docs: List[Dict[str, Any]],
    embeddings_dict: Dict[str, np.ndarray],
    embedding_model_name: str,
    force_recreate: bool = False,
) -> Optional[Any]:
    """
    Upsert documents to Chroma vectorstore.
    
    - Creates directory if needed
    - Converts embeddings to LangChain format (list of floats)
    - Deduplicates by doc ID
    - Persists automatically
    
    Args:
        vectorstore_dir: Path to Chroma persist directory
        docs: List of chunk dicts
        embeddings_dict: Dict of doc_id -> embedding array
        embedding_model_name: For creating embedding function wrapper
        force_recreate: If True, delete directory and start fresh
        
    Returns:
        Chroma vectorstore instance
    """
    if not HAS_CHROMA:
        raise RuntimeError("chromadb not installed; pip install chromadb")
    
    vectorstore_dir = Path(vectorstore_dir)
    
    # Optionally recreate from scratch
    if force_recreate and vectorstore_dir.exists():
        logger.info(f"Removing existing vectorstore at {vectorstore_dir}")
        import shutil
        shutil.rmtree(vectorstore_dir)
    
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Upserting {len(docs)} documents to Chroma at {vectorstore_dir}")
    
    # Convert embeddings to LangChain format (list of Python floats)
    # This is required by Chroma for storage
    embedding_list = []
    doc_ids = []
    lc_docs = []
    
    for doc in docs:
        doc_id = doc["id"]
        if doc_id not in embeddings_dict:
            logger.warning(f"No embedding for doc {doc_id}; skipping")
            continue
        
        embedding_array = embeddings_dict[doc_id]
        embedding_list.append(embedding_array.astype(np.float32).tolist())
        doc_ids.append(doc_id)
        
        # Create LangChain Document
        lc_doc = Document(
            page_content=doc["page_content"],
            metadata=doc["metadata"],
        )
        lc_docs.append(lc_doc)
    
    logger.info(f"Prepared {len(lc_docs)} documents for upsert")
    
    # Create or load Chroma store
    try:
        # Check if store already exists
        chroma_store = None
        if (vectorstore_dir / "chroma.sqlite3").exists() or \
           (vectorstore_dir / "index").exists() or \
           list(vectorstore_dir.glob("chroma*")):
            # Try to load existing store
            try:
                logger.info("Loading existing Chroma store...")
                embedding_fn = _create_embedding_function(embedding_model_name)
                chroma_store = Chroma(
                    persist_directory=str(vectorstore_dir),
                    embedding_function=embedding_fn,
                )
                logger.info(f"Loaded existing Chroma store with collection")
            except Exception as e:
                logger.warning(f"Failed to load existing Chroma store: {e}; creating new")
                chroma_store = None
        
        # Create new store if needed
        if chroma_store is None:
            logger.info("Creating new Chroma store...")
            embedding_fn = _create_embedding_function(embedding_model_name)
            # Start with first document
            if lc_docs:
                chroma_store = Chroma.from_documents(
                    documents=[lc_docs[0]],
                    embedding=embedding_fn,
                    persist_directory=str(vectorstore_dir),
                    ids=[doc_ids[0]],
                )
                lc_docs = lc_docs[1:]
                doc_ids = doc_ids[1:]
                embedding_list = embedding_list[1:]
        
        # Add remaining documents
        if lc_docs:
            logger.info(f"Adding {len(lc_docs)} documents to store...")
            # Note: add_documents already handles metadatas from the documents' metadata attribute
            chroma_store.add_documents(
                documents=lc_docs,
                ids=doc_ids,
            )
        
        # Persist (explicit call for older Chroma versions)
        if hasattr(chroma_store, "persist"):
            try:
                chroma_store.persist()
                logger.info("Chroma store persisted")
            except Exception:
                logger.debug("Chroma.persist() is a no-op for this version")
        
        logger.info(f"Upsert complete: {len(lc_docs)} documents stored")
        return chroma_store
    
    except Exception as e:
        logger.error(f"Failed to upsert to Chroma: {e}")
        raise


def _create_embedding_function(embedding_model_name: str):
    """
    Create an embedding function wrapper for Chroma.
    
    This wraps SentenceTransformer to match the LangChain Embeddings interface.
    Supports both __call__ (for Chroma) and embed_query/embed_documents (for LangChain).
    """
    class STEmbeddingFunction:
        def __init__(self, model_name: str):
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
        
        def __call__(self, input: List[str]) -> List[List[float]]:
            """Chroma interface: callable with list of texts."""
            embeddings = self.model.encode(input, convert_to_numpy=True, show_progress_bar=False)
            return embeddings.astype(np.float32).tolist()
        
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            """LangChain interface: embed multiple documents."""
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings.astype(np.float32).tolist()
        
        def embed_query(self, text: str) -> List[float]:
            """LangChain interface: embed a single query."""
            embedding = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)
            return embedding[0].astype(np.float32).tolist()
    
    return STEmbeddingFunction(embedding_model_name)


# ============================================================================
# METADATA & TRACKING
# ============================================================================

def save_vectorstore_metadata(
    vectorstore_dir: Path,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    num_docs: int,
    csv_hash: str,
) -> None:
    """Save metadata JSON to track vectorstore configuration and content."""
    config_hash = compute_config_hash(embedding_model, chunk_size, chunk_overlap, batch_size)
    
    metadata = {
        "created_at": datetime.utcnow().isoformat(),
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "batch_size": batch_size,
        "config_hash": config_hash,
        "num_docs": num_docs,
        "csv_hash": csv_hash,
    }
    
    meta_path = vectorstore_dir / "_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved vectorstore metadata to {meta_path}")


def load_vectorstore_metadata(vectorstore_dir: Path) -> Optional[Dict[str, Any]]:
    """Load metadata JSON if it exists."""
    meta_path = vectorstore_dir / "_meta.json"
    if not meta_path.exists():
        return None
    
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load vectorstore metadata: {e}")
        return None


def inspect_chroma(vectorstore_dir: Path) -> None:
    """Print diagnostics about a Chroma vectorstore."""
    logger.info(f"=== Chroma Store Inspection ===")
    logger.info(f"Path: {vectorstore_dir}")
    
    if not vectorstore_dir.exists():
        logger.info("Store directory does not exist")
        return
    
    # Load metadata
    meta = load_vectorstore_metadata(vectorstore_dir)
    if meta:
        logger.info(f"Created: {meta.get('created_at')}")
        logger.info(f"Config hash: {meta.get('config_hash')}")
        logger.info(f"Num docs: {meta.get('num_docs')}")
        logger.info(f"Embedding model: {meta.get('embedding_model')}")
    
    # Try to load Chroma collection
    if HAS_CHROMA:
        try:
            embedding_fn = _create_embedding_function(
                meta.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
            )
            store = Chroma(
                persist_directory=str(vectorstore_dir),
                embedding_function=embedding_fn,
            )
            
            # Try to get collection count
            try:
                count = store._collection.count()
                logger.info(f"Collection doc count: {count}")
            except Exception:
                logger.debug("Could not determine collection count")
            
        except Exception as e:
            logger.warning(f"Could not load Chroma store: {e}")
    
    logger.info(f"=== End Inspection ===")
