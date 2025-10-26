# RAG-based SQL generator (example)

This folder contains a small RAG (retrieval-augmented generation) pipeline that
turns natural language queries into safe SQL using business rules written in a
markdown file.

Files added (packaged under `tools/rag/`):
- `tools/rag/embed_rules.py` — load markdown, chunk text, and compute embeddings (sentence-transformers or OpenAI).
- `tools/rag/rag_sql_generator.py` — retrieve top chunks and prompt an LLM to produce SQL.
- `tools/rag/rag_sql_validator.py` — lightweight SQL validation (blocks DROP/DELETE/UPDATE/INSERT/etc.).
- `tools/rag/__init__.py` — package initializer and convenience exports.
- `main.py` — example orchestrator and test harness (imports the package).
Also added small root-level shims (`embed_rules.py`, `rag_sql_generator.py`, `sql_validator.py`) that re-export the package for backward compatibility with older imports.

Quick start
1. (Optional) Create a virtual environment and activate it.

    On Windows cmd.exe:

    python -m venv .venv
    .venv\Scripts\activate

2. Install dependencies.

    pip install -r requirements.txt

   If you want to run locally without OpenAI, install sentence-transformers:

    pip install sentence-transformers

   To use OpenAI for both embeddings and LLM, install and set your API key:

    pip install openai
    setx OPENAI_API_KEY "your_key_here"

3. Ensure your business rules markdown is present. The example looks for
   `TABLE_Chatbot.md` or `TABLE Chatbot.md` in the repo root. The repository
   already contains `TABLE Chatbot.md`.

4. Run the example:

    python main.py

What the pipeline does
- Loads the markdown file containing table definitions and business rules.
- Splits the document into chunks and creates embeddings for each chunk.
- Given a user query, it retrieves the top-k chunks relevant to that query.
- It sends a carefully crafted, strict prompt to an LLM (OpenAI ChatCompletion by default)
  asking for a single SQL statement that uses specific SQL functions and formats.
- The returned SQL is validated by `sql_validator.py` to ensure only safe statements (SELECT/WITH) are allowed.

Extending
- To add new markdown files, call `build_embeddings_from_file(path)` and add the produced chunks/embeddings to the RAG index.
- You can replace OpenAI with another LLM provider by writing a thin adapter in `rag_sql_generator.py`.

Integration with existing workflow
- This repo already contains an LLM adapter at `llm/sql_agent.py`. The RAG pipeline
  implemented here is automatically used by `llm.sql_agent.generate_sql_from_question`
  when the business rules markdown file (`TABLE Chatbot.md`) is present. The agent:
  - Loads or builds embeddings (cached at `cache/rag_embeddings.npz`).
  - Calls the RAG generator (from `tools.rag.rag_sql_generator.RAGSQLGenerator`) to produce SQL that follows strict formatting rules.
  - Falls back to the original prompt-based generation if RAG is unavailable.

To enable or disable RAG explicitly, you can remove or rename the business rules file, or
update `llm/sql_agent.py` to toggle `_HAS_RAG` detection.

Notes and caveats
- The SQL validator here is intentionally lightweight (regex-based). For production you should
  use a proper SQL parser and enforce least privilege at the database level.
- The LLM is instructed to return only SQL. The validator will prevent many dangerous outputs,
  but never trust LLM outputs blindly on production systems.

Contact
If you need help adapting this pipeline to your environment (other LLMs, vector DBs, or executing queries), open an issue or reach out.
