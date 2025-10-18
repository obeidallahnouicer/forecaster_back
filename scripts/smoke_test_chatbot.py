#!/usr/bin/env python3
"""
CLI interface for the RAG chatbot.

Usage:
    python scripts/smoke_test_chatbot.py --query "What are top products?" --top_k 5
    python scripts/smoke_test_chatbot.py --interactive
"""

import argparse
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_chatbot import chatbot


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
        description="RAG Chatbot CLI — Retrieve documents and generate answers"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query to answer"
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode: REPL for multiple queries"
    )
    
    parser.add_argument(
        "--thread_id",
        type=str,
        default="default",
        help="Conversation thread ID (for memory)"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger("rag.chatbot.cli")
    
    logger.info("=" * 80)
    logger.info("RAG Chatbot CLI")
    logger.info("=" * 80)
    
    try:
        # Initialize vectorstore
        logger.info("Initializing vectorstore...")
        store = chatbot.get_vectorstore()
        logger.info("Vectorstore initialized")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return 1
    
    if args.interactive:
        # Interactive mode
        logger.info("Entering interactive mode (type 'quit' to exit)")
        logger.info("=" * 80)
        
        while True:
            try:
                query = input("\nQuery> ").strip()
                
                if query.lower() in ("quit", "exit", "q"):
                    break
                
                if not query:
                    continue
                
                logger.info(f"Processing: {query}")
                
                # Run chat
                result = chatbot.chat(query, thread_id=args.thread_id)
                
                logger.info("=" * 80)
                logger.info(f"Answer ({result['source']}):")
                logger.info(result["answer"])
                logger.info("=" * 80)
                
                # Print metadata
                if "metadata" in result:
                    meta = result["metadata"]
                    logger.info(f"Sources: {', '.join(meta.get('sources', [])[:3])}")
                    logger.info(
                        f"Metadata: model={meta.get('embedding_model')}, "
                        f"chunks={meta.get('num_chunks')}, "
                        f"time={meta.get('retrieval_time_s'):.2f}s"
                    )
                
            except KeyboardInterrupt:
                logger.info("Interrupted")
                break
            except Exception as e:
                logger.exception(f"Error: {e}")
    
    elif args.query:
        # Single query mode
        logger.info(f"Query: {args.query}")
        logger.info("=" * 80)
        
        try:
            result = chatbot.get_answer(args.query, top_k=args.top_k)
            
            logger.info(f"Retrieved {len(result['retrieved_docs'])} documents")
            logger.info("=" * 80)
            logger.info("Top Results:")
            logger.info("=" * 80)
            
            for idx, doc in enumerate(result["retrieved_docs"], 1):
                logger.info(f"\n[{idx}] {doc['id']}")
                logger.info(f"Content: {doc['content']}")
                logger.info(f"Metadata: {doc['metadata']}")
            
            logger.info("=" * 80)
            logger.info("LLM Prompt (for use with OpenAI, Claude, or other LLM):")
            logger.info("=" * 80)
            logger.info(result["llm_prompt"][:500] + "...")
            logger.info("=" * 80)
            logger.info(f"Embedding model: {result['embedding_model']}")
            logger.info(f"Total chunks in store: {result['num_chunks']}")
            logger.info(f"Retrieval time: {result['retrieval_time_s']:.2f}s")
            
        except Exception as e:
            logger.exception(f"Failed to get answer: {e}")
            return 1
    
    else:
        parser.print_help()
        return 0
    
    logger.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
