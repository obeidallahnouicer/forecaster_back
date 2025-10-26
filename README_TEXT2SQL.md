Text2SQL integration (Chat2DB-SQL-7B)
====================================

This folder provides a modular backend skeleton to integrate a local Text->SQL LLM into production.

Key modules
- `text2sql/config.py` - centralized settings
- `text2sql/logger.py` - logging setup
- `text2sql/model_loader.py` - lazy model loading + mock model for tests
- `text2sql/db.py` - SQLAlchemy wrapper for safe SELECT execution
- `text2sql/schema_inspector.py` - small helper to load DB schema
- `text2sql/validator.py` - validate SQL and fuzzy-autocorrect identifiers
- `text2sql/agent.py` - orchestration logic: prompt -> model -> validate -> execute
- `text2sql/api.py` - FastAPI async endpoint (optional)
- `text2sql/cli.py` - simple CLI for batch queries

Tests use the MockModel to avoid heavy model loads and run against in-memory SQLite.

Usage
-----
- Run the FastAPI app: `uvicorn text2sql.api:app --reload`
- Use the CLI: `python -m text2sql.cli --query "How many singers do we have?"`

Notes
-----
- The real model is loaded lazily; set `MODEL_USE_MOCK=1` to force the mock model.
- For production/GPU, verify torch/transformers and consider quantization options.
