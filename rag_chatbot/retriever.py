from typing import Optional, Iterable, List, Any
import pandas as pd
import logging
from pathlib import Path
import hashlib
import json
from datetime import datetime

try:
    # Prefer community Chroma (non-deprecated)
    from langchain_community.vectorstores import Chroma
    _HAS_CHROMA = True
except ImportError:
    try:
        # Fallback to deprecated import (will show warning)
        from langchain.vectorstores import Chroma
        _HAS_CHROMA = True
    except ImportError:
        Chroma = None
        _HAS_CHROMA = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    SentenceTransformer = None
    _HAS_ST = False

# Import the new indexer module for robust pipeline
try:
    from . import indexer as rag_indexer
    _HAS_INDEXER = True
except ImportError:
    _HAS_INDEXER = False
from langchain.schema import Document
from . import config, data_loader
import os

logger = logging.getLogger("rag.retriever")


def _load_cached_vectorstore() -> Optional[Any]:
    """Try to load an existing persisted Chroma store without re-embedding."""
    logger.info("Attempting to load existing Chroma vectorstore...")
    
    # Check if persisted store exists
    chroma_dir = Path(config.VECTORSTORE_DIR) / "chroma"
    if not chroma_dir.exists():
        logger.info("No persisted vectorstore found at %s", chroma_dir)
        return None
    
    if not _HAS_CHROMA:
        logger.error("Chroma not available; cannot load vectorstore")
        return None
    
    try:
        # Create embedding function using the new indexer
        if _HAS_INDEXER:
            embedding_fn = rag_indexer._create_embedding_function(config.EMBEDDING_MODEL)
        else:
            logger.error("indexer module not available; cannot create embeddings")
            return None
        
        # Load existing store
        chroma_store = Chroma(
            persist_directory=str(chroma_dir),
            embedding_function=embedding_fn,
        )
        
        # Log collection stats
        try:
            count = chroma_store._collection.count()
            logger.info("Loaded existing Chroma store with %d documents", count)
        except Exception:
            logger.info("Loaded existing Chroma store")
        
        return chroma_store
    
    except Exception as e:
        logger.exception("Failed to load existing Chroma store: %s", e)
        return None


def build_vectorstore(recreate: bool = False, df: Optional[pd.DataFrame] = None):
    """
    Build or load a Chroma vectorstore from CSV data.
    
    Robust, resumable implementation using the new indexer pipeline.
    
    Args:
        recreate: If True, delete existing store and rebuild from scratch
        df: Optional DataFrame to use; otherwise loads from config.DATA_PATH
        
    Returns:
        Chroma vectorstore instance, or None on failure
        
    Behavior:
    - If recreate=False (default): tries to load existing persisted store first
    - If store doesn't exist or recreate=True: rebuilds by chunking, embedding, and upserting
    - Uses deterministic doc IDs for deduplication
    - Saves metadata for config tracking and resumability
    """
    
    logger.info("build_vectorstore called (recreate=%s)", recreate)
    
    if not _HAS_CHROMA:
        logger.error("Chroma not installed; cannot build vectorstore")
        return None
    
    if not _HAS_INDEXER:
        logger.error("rag.indexer module not available; cannot build vectorstore")
        return None
    
    chroma_dir = Path(config.VECTORSTORE_DIR) / "chroma"
    
    # Try to load existing store (unless recreate requested)
    if not recreate:
        store = _load_cached_vectorstore()
        if store is not None:
            logger.info("Returning existing vectorstore (recreate=False)")
            return store
    
    # Need to rebuild
    logger.info("Building vectorstore from scratch (recreate=%s)", recreate)
    
    try:
        # Load data
        if df is None:
            logger.info("Loading data from %s", config.DATA_PATH)
            df = data_loader.load_and_preprocess()
        else:
            logger.info("Using provided DataFrame (%d rows)", len(df))
        
        # Chunk texts
        chunks = rag_indexer.chunk_texts(
            df,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            max_rows=None,
        )
        logger.info("Created %d chunks", len(chunks))
        
        # Compute embeddings
        embeddings_dict = rag_indexer.compute_embeddings_in_batches(
            chunks,
            embedding_model_name=config.EMBEDDING_MODEL,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            device=None,  # Auto-detect
            force_embed=False,
        )
        logger.info("Computed embeddings for %d documents", len(embeddings_dict))
        
        # Upsert to Chroma
        store = rag_indexer.upsert_to_chroma(
            chroma_dir,
            chunks,
            embeddings_dict,
            config.EMBEDDING_MODEL,
            force_recreate=recreate,
        )
        
        # Save metadata
        csv_hash = rag_indexer.compute_csv_hash(df)
        rag_indexer.save_vectorstore_metadata(
            chroma_dir,
            config.EMBEDDING_MODEL,
            config.CHUNK_SIZE,
            config.CHUNK_OVERLAP,
            config.EMBEDDING_BATCH_SIZE,
            len(chunks),
            csv_hash,
        )
        
        logger.info("Vectorstore built successfully at %s", chroma_dir)
        return store
    
    except Exception as e:
        logger.exception("Failed to build vectorstore: %s", e)
        return None


def get_retriever(k: int = 5, recreate: bool = False):
    """
    Get a LangChain retriever backed by Chroma.
    
    Args:
        k: Number of documents to retrieve
        recreate: If True, rebuild vectorstore from scratch
        
    Returns:
        LangChain Retriever instance
        
    Raises:
        RuntimeError: If vectorstore cannot be built or loaded
    """
    store = build_vectorstore(recreate=recreate)
    if store is None:
        logger.error("Vectorstore not available after build_vectorstore(recreate=%s)", recreate)
        raise RuntimeError(
            "Vectorstore not available. Run 'python scripts/build_vectorstore.py --recreate' "
            "to index data, or check that chromadb is installed and embeddings are available."
        )
    
    logger.info("Returning retriever with k=%d", k)
    return store.as_retriever(search_kwargs={"k": k})
