#!/usr/bin/env python3
"""
Smoke test: query the persisted Chroma vectorstore and print results.

Usage:
    python scripts/smoke_test_query.py --query "What are main risks?" --k 3
    python scripts/smoke_test_query.py --vectorstore-dir cache/vectorstore/chroma --k 5
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_chatbot import indexer, config


def setup_logging():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)


def main():
    parser = argparse.ArgumentParser(
        description="Query the persisted Chroma vectorstore and display top-K results"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What are the top selling products?",
        help="Query string to search for"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of top results to display"
    )
    parser.add_argument(
        "--vectorstore-dir",
        type=str,
        default=None,
        help="Path to Chroma directory (default: cache/vectorstore/chroma)"
    )
    
    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger("rag.smoke_test")
    
    logger.info("=" * 80)
    logger.info("Chroma Vectorstore Smoke Test")
    logger.info("=" * 80)
    
    # Determine vectorstore dir
    if args.vectorstore_dir:
        vectorstore_dir = Path(args.vectorstore_dir)
    else:
        vectorstore_dir = config.VECTORSTORE_DIR / "chroma"
    
    logger.info(f"Vectorstore dir: {vectorstore_dir}")
    
    if not vectorstore_dir.exists():
        logger.error(f"Vectorstore directory not found at {vectorstore_dir}")
        logger.error("Run 'python scripts/build_vectorstore.py' first")
        return 1
    
    try:
        # Load metadata
        meta = indexer.load_vectorstore_metadata(vectorstore_dir)
        if meta:
            logger.info(f"Vectorstore config hash: {meta.get('config_hash')}")
            logger.info(f"Total docs: {meta.get('num_docs')}")
            logger.info(f"Embedding model: {meta.get('embedding_model')}")
        
        # Load vectorstore
        logger.info(f"Loading Chroma store from {vectorstore_dir}")
        embedding_model = meta.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2") if meta else config.EMBEDDING_MODEL
        embedding_fn = indexer._create_embedding_function(embedding_model)
        
        from langchain_community.vectorstores import Chroma
        store = Chroma(
            persist_directory=str(vectorstore_dir),
            embedding_function=embedding_fn,
        )
        
        # Try to get collection size
        try:
            count = store._collection.count()
            logger.info(f"Collection has {count} documents")
        except Exception as e:
            logger.debug(f"Could not determine collection size: {e}")
        
        # Run query
        logger.info(f"=" * 80)
        logger.info(f"Query: {args.query}")
        logger.info(f"Retrieving top {args.k} documents...")
        logger.info(f"=" * 80)
        
        # Search with scores
        results = store.similarity_search_with_score(args.query, k=args.k)
        
        if not results:
            logger.warning("No results found")
            return 0
        
        logger.info(f"\nTop {args.k} results:\n")
        
        for idx, (doc, score) in enumerate(results, 1):
            logger.info(f"--- Result #{idx} (score: {score:.4f}) ---")
            logger.info(f"Content:\n{doc.page_content[:200]}{'...' if len(doc.page_content) > 200 else ''}")
            logger.info(f"Metadata: {doc.metadata}\n")
        
        # Compose a sample prompt for LLM
        logger.info(f"=" * 80)
        logger.info("Sample LLM prompt composition:")
        logger.info(f"=" * 80)
        
        context_docs = "\n\n".join([
            f"[Document {idx}]:\n{doc.page_content}"
            for idx, (doc, _) in enumerate(results, 1)
        ])
        
        prompt = f"""You are an expert analyst. Use ONLY the following retrieved documents (do not hallucinate).
For each claim, append a source citation in square brackets with the doc id.

Context:
{context_docs}

Question:
{args.query}

Deliverable:

Short answer / recommendation (2-4 sentences)

Short reasoning steps with citations

List of documents used (id)
If answer not supported, say "I don't know based on the provided documents."
"""
        
        logger.info(prompt)
        logger.info(f"=" * 80)
        logger.info("SUCCESS: Vectorstore query test passed")
        logger.info(f"=" * 80)
        
        return 0
    
    except Exception as e:
        logger.exception(f"FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
