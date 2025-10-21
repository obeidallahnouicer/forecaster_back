"""
Configuration module for RAG chatbot.

All settings can be overridden via environment variables.
See DEV_NOTES.md for more information.
"""

from pathlib import Path
import os
from dotenv import load_dotenv
import hashlib
import json
import logging

# Load .env from project root if present
root = Path(__file__).resolve().parent.parent
env_path = root / ".env"
if env_path.exists():
    load_dotenv(env_path)

logger = logging.getLogger("rag.config")

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

# Groq API key (accept common env var names)
GROQ_API_KEY = os.getenv("Groq_Api_key") or os.getenv("GROQ_API_KEY")
# Groq model name
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Multi-agent orchestration flag: when True, run the Retriever->Analysis->Reasoning->Advisor->Validator pipeline
MULTI_AGENT_ORCHESTRATION = os.getenv("MULTI_AGENT_ORCHESTRATION", "true").lower() in ("1", "true", "yes")

# ============================================================================
# EMBEDDING CONFIGURATION (Production-ready)
# ============================================================================

# Primary embedding model: Jina v3 (state-of-the-art production model)
# Override with EMBEDDING_MODEL env var or JINA_EMBEDDING_MODEL for backward compat
EMBEDDING_MODEL = os.getenv("JINA_EMBEDDING_MODEL") or os.getenv(
    "EMBEDDING_MODEL", "jinaai/jina-embeddings-v3"
)

# Device: auto-detect GPU if available, else CPU
def _get_embedding_device() -> str:
    """Detect GPU; fallback to CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"

EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", _get_embedding_device())

# ============================================================================
# DATA PATHS
# ============================================================================

# Primary data source: forecast-summary.csv at repo root
DATA_PATH = Path(__file__).resolve().parent.parent / "cache" / "forecasts"
# When True, force the RAG/chatbot to use the repository-level 'forecast-summary.csv'
# and do not fallback to uploaded server_data files. Controlled via env var
# RAG_FORCE_REPO_SUMMARY (default: 0). Set to '1' to enable.
FORCE_REPO_SUMMARY = os.getenv("RAG_FORCE_REPO_SUMMARY", "0") in ("1", "true", "True")

# Data-backed deterministic answers are permanently disabled in this build.
# This enforces a strict policy: the system must not produce deterministic
# dataset-only answers or structured fallbacks. Leave as False to make the
# behavior unconditional (no environment toggle).
ENABLE_DATA_BACKED_ANSWERS = False

# Vectorstore persistence directory
VECTORSTORE_DIR = Path(__file__).resolve().parent.parent / "cache" / "vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# Legacy FAISS constants (deprecated; kept for backward compatibility only)
FAISS_INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
METADATA_PATH = VECTORSTORE_DIR / "metadatas.pkl"

# Memory persistence directory
MEMORY_DIR = Path(__file__).resolve().parent.parent / "cache" / "memories"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CHUNKING & EMBEDDING PARAMETERS
# ============================================================================

# Text chunking: size of each text chunk (in characters)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 2048))

# Text chunking: overlap between consecutive chunks (in characters)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 128))

# Embedding batch size (tune for your CPU/GPU; larger for GPU)
# Default is conservative for CPU; adjust if you have GPU available
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 64))

# ============================================================================
# RETRIEVAL PARAMETERS
# ============================================================================

# Default number of documents to retrieve per query
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 5))

# ============================================================================
# FASTAPI CONFIGURATION
# ============================================================================

# API host and port
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# ============================================================================
# CONFIG HASH & RESUMABILITY
# ============================================================================

def compute_embedding_config_hash() -> str:
    """
    Compute a deterministic hash of the embedding configuration.
    
    This is used to detect when config changes and embeddings must be recomputed.
    
    Returns:
        16-character hex string (SHA256)
    """
    config_dict = {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
    }
    config_json = json.dumps(config_dict, sort_keys=True)
    hash_obj = hashlib.sha256(config_json.encode())
    return hash_obj.hexdigest()[:16]


# ============================================================================
# DEBUG & LOGGING
# ============================================================================

# Log configuration on import (useful for debugging)
if os.getenv("RAG_CONFIG_DEBUG", "0") == "1":
    logger.info(f"Embedding model: {EMBEDDING_MODEL}")
    logger.info(f"Embedding device: {EMBEDDING_DEVICE}")
    logger.info(f"Chunk size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}")
    logger.info(f"Config hash: {compute_embedding_config_hash()}")
