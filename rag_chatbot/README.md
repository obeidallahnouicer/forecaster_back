RAG Chatbot
=============

This package provides a Retrieval-Augmented Generation chatbot using LangChain, OpenAI and FastAPI.

How to run
----------

1. Create a .env with OPENAI_API_KEY
2. Install dependencies from requirements.txt
3. Run with uvicorn:

uvicorn rag_chatbot.api:app --host 0.0.0.0 --port 8000

Embeddings and models
---------------------

This project uses a sentence-transformers style embedding model by default. The embedding model can be set via
the `EMBEDDING_MODEL` environment variable (see `rag_chatbot/config.py`). The default is `jinaai/jina-embeddings-v3`
and the retriever will load it using `sentence-transformers` with `trust_remote_code=True` so no Hugging Face API
token is required.

