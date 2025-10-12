from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env from project root if present
root = Path(__file__).resolve().parent.parent
env_path = root / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Groq-only configuration (no OpenAI)
# Groq API key (accept common env var names)
GROQ_API_KEY = os.getenv("Groq_Api_key") or os.getenv("GROQ_API_KEY")
# Groq model name
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# Embeddings (Hugging Face sentence-transformers)
HUGGINGFACE_EMBEDDING_MODEL = os.getenv(
    "HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Data and cache paths
DATA_PATH = Path(__file__).resolve().parent.parent / "chatbotdf.csv"
VECTORSTORE_DIR = Path(__file__).resolve().parent.parent / "cache" / "vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# FAISS index file name
FAISS_INDEX_PATH = VECTORSTORE_DIR / "faiss_index.faiss"
METADATA_PATH = VECTORSTORE_DIR / "metadatas.pkl"

# Memory persistence directory
MEMORY_DIR = Path(__file__).resolve().parent.parent / "cache" / "memories"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# Retriever settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 2048))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 128))

# Embedding batch size (tune for your CPU/GPU)
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 128))

# FastAPI settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
