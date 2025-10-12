RAG Chatbot
=============

This package provides a Retrieval-Augmented Generation chatbot using LangChain, OpenAI and FastAPI.

How to run
----------

1. Create a .env with OPENAI_API_KEY
2. Install dependencies from requirements.txt
3. Run with uvicorn:

uvicorn rag_chatbot.api:app --host 0.0.0.0 --port 8000

Using Hugging Face (OpenAI OSS)
--------------------------------

This project is configured to use Hugging Face models exclusively when `HUGGINGFACEHUB_API_TOKEN` and
`HUGGINGFACE_LLM` are set in your `.env` file. Set these values in the project root `.env` file. Example:

HUGGINGFACEHUB_API_TOKEN=hf_xxxYOURTOKENxxx
HUGGINGFACE_LLM=tiiuae/falcon-7b-instruct  # example; for OpenAI OSS use the HF repo id you want
HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

Notes:
- `HUGGINGFACE_LLM` must be the Hugging Face repo id for the model you want to use (OpenAI OSS or other HF-hosted models).
- `HUGGINGFACE_EMBEDDING_MODEL` should point to a sentence-transformers model or another embedding model available on HF.

