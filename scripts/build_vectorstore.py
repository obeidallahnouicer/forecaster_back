#!/usr/bin/env python3
"""
CLI tool to build and manage the Chroma vectorstore for RAG.

Usage:
    python scripts/build_vectorstore.py --help
    python scripts/build_vectorstore.py --dev --max-rows 50
    python scripts/build_vectorstore.py --recreate
    python scripts/build_vectorstore.py --force-embed --batch-size 32
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add parent dir to path so we can import from rag_chatbot
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_chatbot import indexer, config
import pandas as pd


def setup_logging(level=logging.INFO):
    """Configure logging with timestamps and colors."""
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    
    # Configure root logger
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    
    # Suppress verbose logs from third-party libraries
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(
        description="Build and manage Chroma vectorstore for RAG indexing"
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV to index (default: forecaster_back/forecast-summary.csv)"
    )
    
    parser.add_argument(
        "--vectorstore-dir",
        type=str,
        default=None,
        help="Path to Chroma persist directory (default: cache/vectorstore)"
    )
    
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete existing vectorstore and rebuild from scratch"
    )
    
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use small dev embedding model (all-MiniLM-L6-v2) instead of Jina"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding computation (default: 64)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Chunk size for text splitting (default: 2048)"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=128,
        help="Chunk overlap in characters (default: 128)"
    )
    
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit to first N rows of CSV (useful for testing)"
    )
    
    parser.add_argument(
        "--force-embed",
        action="store_true",
        help="Force re-embedding of all documents (even if they exist)"
    )
    
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Force CPU or CUDA device (auto-detect if not specified)"
    )
    
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Just inspect existing vectorstore and exit"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger("rag.build_vectorstore")
    
    logger.info("=" * 80)
    logger.info("RAG Vectorstore Builder")
    logger.info("=" * 80)
    
    # Determine CSV path
    if args.csv:
        csv_path = Path(args.csv)
    else:
        # Default: look for forecast-summary.csv at repo root
        repo_root = Path(__file__).resolve().parent.parent
        csv_path = repo_root / "forecast-summary.csv"
        if not csv_path.exists():
            csv_path = repo_root / "cache" / "forecasts" / "summary.csv"
    
    logger.info(f"CSV path: {csv_path}")
    
    # Determine vectorstore dir
    if args.vectorstore_dir:
        vectorstore_dir = Path(args.vectorstore_dir)
    else:
        vectorstore_dir = config.VECTORSTORE_DIR / "chroma"
    
    logger.info(f"Vectorstore dir: {vectorstore_dir}")
    
    # Handle --inspect
    if args.inspect:
        indexer.inspect_chroma(vectorstore_dir)
        return 0
    
    # Select embedding model
    if args.dev:
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info("DEV MODE: using small embedding model")
    else:
        embedding_model = "jinaai/jina-embeddings-v3"
        logger.info("PRODUCTION MODE: using Jina embeddings v3")
    
    # Select device
    if args.device:
        device = args.device
    else:
        device = indexer.get_embedding_device()
    
    try:
        start_time = datetime.now()
        
        # Load CSV
        logger.info(f"Loading CSV from {csv_path}")
        df = indexer.load_and_normalize_csv(csv_path)
        logger.info(f"INFO:rag.indexer:Loaded CSV: {len(df)} rows, {len(df.columns)} columns")
        
        if args.max_rows:
            df = df.head(args.max_rows)
            logger.info(f"Limiting to {args.max_rows} rows for testing")
        
        # Compute CSV hash before chunking
        csv_hash = indexer.compute_csv_hash(df)
        logger.info(f"CSV content hash: {csv_hash}")
        
        # Chunk texts
        logger.info(f"INFO:rag.indexer:Chunking texts (size={args.chunk_size}, overlap={args.chunk_overlap})")
        chunks = indexer.chunk_texts(
            df,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_rows=None,  # Already limited above
        )
        logger.info(f"INFO:rag.indexer:Created {len(chunks)} chunks from {len(df)} rows")
        
        # Compute embeddings
        logger.info(f"INFO:rag.indexer:Computing embeddings with batch_size={args.batch_size}")
        embeddings_dict = indexer.compute_embeddings_in_batches(
            chunks,
            embedding_model_name=embedding_model,
            batch_size=args.batch_size,
            device=device,
            force_embed=args.force_embed,
        )
        logger.info(f"INFO:rag.indexer:Computed {len(embeddings_dict)} embeddings")
        
        # Upsert to Chroma
        logger.info(f"INFO:rag.indexer:Upserting to Chroma")
        vectorstore = indexer.upsert_to_chroma(
            vectorstore_dir,
            chunks,
            embeddings_dict,
            embedding_model,
            force_recreate=args.recreate,
        )
        
        # Save metadata
        indexer.save_vectorstore_metadata(
            vectorstore_dir,
            embedding_model,
            args.chunk_size,
            args.chunk_overlap,
            args.batch_size,
            len(chunks),
            csv_hash,
        )
        logger.info(f"INFO:rag.indexer:Vectorstore persisted at {vectorstore_dir}")
        
        # Final inspection
        indexer.inspect_chroma(vectorstore_dir)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("=" * 80)
        logger.info(f"SUCCESS: Vectorstore built in {elapsed:.1f} seconds")
        logger.info("=" * 80)
        
        return 0
    
    except Exception as e:
        logger.exception(f"ERROR:rag.indexer:Failed to build vectorstore: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
