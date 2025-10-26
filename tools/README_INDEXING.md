Indexing and retrieval for domain MD (TABLE Chatbot.md)

Overview
--------
This small toolkit creates chunked documents from `TABLE Chatbot.md`, generates embeddings (via OpenAI),
and provides a simple file-based retriever to find the most relevant chunks for a query.

Files added
- `tools/index_md.py` — split the markdown into sections and chunks; write `chunks.jsonl`.
- `tools/embeddings.py` — create embeddings for chunks and write `vectors.jsonl`.
- `tools/retriever_example.py` — minimal offline retriever using cosine similarity.

Quick start
-----------
1. Install dependencies (openai and numpy if you don't already have them):

   pip install openai numpy

2. Create chunks from the provided MD (run from repo root):

   python -m tools.index_md --input "TABLE Chatbot.md" --out chunks.jsonl --chunk-size 800

3. Export embeddings (requires OPENAI_API_KEY in environment):

   set OPENAI_API_KEY=your_key_here
   python -m tools.embeddings --in chunks.jsonl --out vectors.jsonl --model text-embedding-3-small

4. Run a retrieval example:

   python -m tools.retriever_example --vectors vectors.jsonl --query "Which clients are inactive since 3 months?" --topk 4

How to integrate into query flow
--------------------------------
At runtime (when a user asks a question):

1. Embed the user query with the same embedding model.
2. Use cosine similarity to find the top N chunks from `vectors.jsonl`.
3. Inject those chunk texts (and the business rules portion) into the LLM prompt as context.
4. Ask the LLM to generate SQL or textual insights using that context.

Notes & next steps
- The current implementation uses OpenAI's Python package. If you prefer a different provider, adapt `tools/embeddings.py` and `tools/retriever_example.py`.
- Consider storing vectors in a vector DB (FAISS, Milvus, Pinecone, etc.) for scale.
- Add tests for chunking and retrieval (todo).
