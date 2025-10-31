# Sales Forecaster & Business Intelligence API# Sales Forecaster & Business Intelligence API - LangChain + Guardrails Edition



**Version 5.0.0** - Strict Validation Pipeline with LangChain + Guardrails AI**Version 4.0.0** - Production-ready with LangChain orchestration and Guardrails AI safety



## 🎯 Overview## 🎯 Overview



Production-ready FastAPI backend featuring a **SQL chatbot with strict validation** and **sales forecasting**.A FastAPI-based business intelligence system with two core capabilities:



**Core Capabilities**:1. **Text-to-SQL Chatbot** - Natural language to SQL using LangChain chains with Guardrails AI safety

1. **SQL Chatbot** - Natural language → Validated SQL → Insights (LangChain + Guardrails AI)2. **Sales Forecasting** - Upload sales data, generate forecasts with multiple ML models

2. **Sales Forecasting** - Time series predictions with Prophet

**Architecture**: LangChain + Groq (LLaMA 3.3 70B) + Guardrails AI for safety

**Technology Stack**:

- **LangChain**: Orchestration and prompt management**Key Features**:

- **Groq**: Llama 3.3 70B inference (fast, scalable)- 🔗 **LangChain Integration**: Modular prompt templates and composable chains

- **Guardrails AI**: Input/output validation for safety- 🛡️ **Guardrails AI**: SQL injection prevention, PII detection/masking, syntax validation

- **FastAPI**: High-performance API framework- 🚀 **Groq LLM**: Fast inference with LLaMA 3.3 70B

- 📊 **Data-Driven**: Direct SQL on structured data (no RAG overhead)

---

---

## 🛡️ Strict Validation Pipeline

## 🚀 Quick Start

Every user query flows through a **5-stage validation pipeline** with **NO bypass logic**:

### Installation

```

┌────────────────────────────────────────────────────────────────┐```bash

│  STAGE 1: INPUT VALIDATION (PII Detector)                      │# Install dependencies

│  ────────────────────────────────────────────────────────────  │pip install -r requirements.txt

│  Purpose: Detect PII/sensitive data in user input              │

│  Action:  ✓ PASS → Continue to LLM                             │# Set Groq API key

│           ✗ FAIL → REJECT immediately (log & return error)     │export GROQ_API_KEY=your_api_key_here

# Sales Forecaster & Business Intelligence (forecaster_back)

Comprehensive documentation for the Sales Forecaster & Business Intelligence codebase.

This README documents the project's purpose, architecture, features, important files and packages, how to run the app and the dashboard, API endpoints, internal modules (agents, core, cache, llm), caveats, troubleshooting and suggested next steps.

## Table of contents

- Project summary
- Key features
- Architecture overview
- Important files and directories
- Installation & setup
- Environment variables and configuration
- Running the FastAPI server
- Running the Streamlit dashboard
- CLI utilities and examples
- Main modules (description + responsibilities)
  - `main.py` (FastAPI entrypoint)
  - `streamlit_app.py` (dashboard)
  - `sales_forecaster.py` (forecast engine)
  - `agents/` (QueryAgent, ExecutorAgent, etc.)
  - `core/` (config, db connection, schema loader)
  - `cache/` (cache managers)
  - `llm/` (LLM client and adapter)
  - `api/` (routers)
- APIs (endpoints summary)
- Data formats & expectations
- Testing
- Troubleshooting & notes
- Contribution & roadmap

---

## Project summary

This repository implements a unified backend for sales forecasting and a Text-to-SQL chatbot. It combines the following capabilities:

- Multi-method sales forecasting (SMA, Exponential Smoothing, Linear Regression, ARIMA, Prophet, XGBoost/RandomForest)
- Per-article and summary forecast caching with intelligent invalidation
- Text-to-SQL pipeline: natural language -> LLM -> SQL -> Guardrails validation -> execution (in-memory SQLite)
- Streamlit-based forecasting dashboard (interactive UI, charts, CSV upload)
- FastAPI-based REST API (SQL Chat, Forecasts, Dashboard endpoints)
- Groq (Llama 3.3 70B) adapter support (via `llm.client`) with rate-limiting behavior
- PII detection and SQL validation guardrails for safer LLM outputs

Intended as a single-user / prototype system that can be extended for production environments.

---

## Key features

- Text-to-SQL with a strict validation pipeline (PII detection, SQL validation, identifier checks, LLM retry/correction)
- In-memory DB loader for local CSV/XLSX files so SQL queries can be executed without external DB
- Forecast engine with caching (per-article cache files + a ForecastCacheManager for a more advanced cache)
- Streamlit dashboard with interactive charts and method comparison per article
- API endpoints for programmatic access and integration
- Rate-limited LLM client with cooldown persistence

---

## Architecture overview

High-level components:

- API layer: `main.py` creates a FastAPI app and includes routers under `api/routers`.
- Agents: `agents/query_agent.py` (QueryAgent) handles NL -> SQL generation + validation; `agents/executor_agent.py` executes validated SQL.
- Core: configuration, DB connection helpers (in-memory SQLite), table/schema helpers.
- Cache: local file-based cache managers to store forecasts, uploads, and summary results.
- LLM: lightweight adapter to call Groq (Llama 3.3 70B) or fallback behavior for unconfigured LLM keys.
- Streamlit dashboard: `streamlit_app.py` uses `SalesForecaster` to load data and show forecasts.

Flow for Text-to-SQL:

1. Client asks a natural-language question.
2. `QueryAgent` runs input PII checks and sends a prompt to the LLM adapter.
3. The produced SQL is cleaned and validated by `SQLOutputValidator` (guardrails).
4. Identifier existence checks and LLM retry path attempt to fix unknown identifier issues.
5. When validated, `ExecutorAgent` executes the SQL using `core.db_connection.execute_select`.
6. Results are returned to the client.

Flow for forecasting (Streamlit / API):

1. User uploads dataset (CSV/Excel) or uses preloaded CSVs in repo.
2. `SalesForecaster` normalizes columns and prepares data (yearly/monthly aggregation).
3. Per-article forecasts are computed using several methods and cached per article.
4. Aggregate summary forecasts are saved to a summary cache (parquet) for fast retrieval.

---

## Important files and directories (quick map)

- `main.py` - FastAPI application entrypoint, CORS, middleware, docs and router inclusion.
- `streamlit_app.py` - Streamlit dashboard UI and orchestration for the `SalesForecaster`.
- `sales_forecaster.py` - Core forecasting engine. Implements many forecasting methods, caching, and utilities.
- `agents/` - Agent modules:
  - `agents/query_agent.py` - Legacy QueryAgent implementing the NL->SQL pipeline with validation/retries.
  - `agents/executor_agent.py` - Executes validated SQL queries and returns results.
- `core/` - Core utilities:
  - `core/config.py` - Centralized configuration (env var overrides).
  - `core/db_connection.py` - In-memory sqlite loader & SQL execution helper.
  - `core/schema_loader.py`, `core/table_schemas.py` - schema helpers used by the QueryAgent for identifier checks and to present the schema to the LLM.
- `cache/` - Various caching modules and an advanced `ForecastCacheManager` and docs.
  - `cache/forecast_cache.py` - Forecast cache manager with intelligent invalidation.
  - `cache/README.md` - Design doc describing the production-grade caching system.
- `llm/` - LLM adapter code:
  - `llm/client.py` - Groq adapter with rate-limiting and cooldowns. If `GROQ_API_KEY` is not configured, it returns a helpful fallback response.
  - The project uses an LLM adapter pattern (e.g., `llm.sql_agent`) to centralize how prompts are constructed and how model outputs are parsed.
- `api/routers/` - FastAPI routers (e.g., `sql_chat`, `forecasts`, `dashboard`).
- `prompts/` - Prompt templates and helper functions.
- `tools/` - RAG & embedding utilities used by `main.py` example harness.
- `tests/` - Tests (unit tests for key components if present; run with pytest).

---

## Installation & setup

This project uses Python (3.10+ recommended). Core dependencies are listed in `requirements.txt`.

1. Create and activate a virtual environment (Windows PowerShell example):

   ```powershell
   python -m venv .venv; .\.venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. (Optional) Create a `.env` in the project root to override default configuration (see `core/config.py`).

3. If you plan to use Groq (Llama 3.3 70B) set the environment variable `GROQ_API_KEY` or `Groq_Api_key`.

Note: Several forecasting backends are optional and only used if installed:
- `statsmodels` (ARIMA), `prophet` (Prophet), `xgboost` (XGBoost) — `sales_forecaster.py` checks for availability and toggles features accordingly.

---

## Environment variables and configuration

Important env vars reflected in `core/config.py` (defaults shown in code):

- `HOST` (default `0.0.0.0`) - server host
- `PORT` (default `8000`) - server port
- `DEBUG` - debug mode `true/false`
- `GROQ_API_KEY` or `Groq_Api_key` - Groq API key
- `GROQ_MODEL` - default LLM model (e.g., `llama-3.3-70b-versatile`)
- `LLM_RETRY_ATTEMPTS` - number of LLM retries for SQL repairs
- `LOG_LEVEL`, `LOG_FORMAT` - logging config
- `MAX_FILE_SIZE` - file upload limit

Many of these can be set in a `.env` file located at the project root.

---

## Running the FastAPI server (HTTP API)

`main.py` constructs the FastAPI app and runs with Uvicorn when executed directly. To run the server for development:

- Using python directly:

  ```powershell
  python main.py
  ```

  This will start Uvicorn with the configured `HOST` and `PORT` from `core/config.py`.

- Or run Uvicorn explicitly for auto-reloading during development:

  ```powershell
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
  ```

Once running, API docs are available at `http://<host>:<port>/docs`.

---

## Running the Streamlit dashboard

The Streamlit dashboard is implemented in `streamlit_app.py` and uses the `SalesForecaster` directly.

To run:

```powershell
streamlit run streamlit_app.py
```

The dashboard allows uploading CSV/Excel sales data, selecting forecasting frequency (`yearly` or `monthly`) and running forecasts either per-article or for the whole dataset. It also provides cache management tools.

---

## CLI utilities & example harness

`main.py` also contains an example harness (at top of file) that demonstrates building embeddings from a business rules markdown and running a RAG SQL generation example. That example uses utilities in `tools/rag/*` and is intended as a test harness, not a production script.

---

## Main modules (detailed)

### `main.py` (FastAPI entrypoint)

- Builds a FastAPI application with metadata and middleware.
- Configures logging using `core.config` settings.
- Includes routers from `api.routers` (e.g., `sql_chat`, `forecasts`, `dashboard`).
- Adds `/health` and `/` root endpoints for basic system information.

### `streamlit_app.py` (dashboard)

- Interactive dashboard using `SalesForecaster` to load uploaded files, compute forecasts and show charts.
- Supports monthly/yearly aggregation, several advanced forecast parameters, cache clearing, and CSV export.
- Visualizations implemented with Plotly (time series, comparison, metrics charts).

Key features:
- File upload (CSV/XLSX)
- Frequency selection: `yearly` / `monthly`
- Forecast-all and single-article forecast flows
- Cache management (clear per session)
- Download results as CSV

### `sales_forecaster.py` (forecast engine)

Responsibilities:
- Data normalization and flexible column mapping (handles many CSV variants).
- `clean_data()` and `prepare_data()` for aggregation by article and period.
- Multiple forecasting methods with metrics support:
  - `simple_moving_average` (SMA)
  - `exponential_smoothing`
  - `linear_regression_forecast`
  - `arima_forecast` (optional: requires statsmodels)
  - `prophet_forecast` (optional: requires prophet)
  - `xgboost_forecast` (XGBoost or fallback to RandomForest)
- Per-article caching (CSV cache files under `cache/` organized by source hash and frequency)
- `forecast_article()` returns a consistent dictionary shape usable by both the Streamlit UI and API consumers, including serialized history fields and metrics.
- `forecast_all_articles()` computes batch forecasts and writes `summary.parquet` for quick retrieval.

Design notes:
- Fast-mode heuristics for short series to avoid heavy models.
- Data fingerprinting used for cache isolation; `model_version` used for invalidation.

### `agents/`

- `agents/query_agent.py` implements the natural language -> SQL pipeline. Key responsibilities:
  - Input PII validation (`PIIInputValidator`)
  - Delegate LLM generation to `llm.sql_agent` adapter and parse response
  - Clean and parameterize SQL (replace inline INTERVALs and dates with named parameters)
  - Validate SQL via `SQLOutputValidator` (Guardrails)
  - Verify identifiers (tables/columns) exist in `core.schema_loader` snapshot
  - Attempt auto-fix using fuzzy matching and limited LLM retry attempts if unknown identifiers are found
  - Execute query using `core.db_connection.execute_select` after mapping logical to physical table names

- `agents/executor_agent.py` executes validated SQL. It expects the `validation_passed` flag to be True and returns rows/columns/rowcount and execution time. It uses `core.db_connection.execute_select`.

### `core/`

- `core/config.py` centralizes configuration and environment variable defaults.
- `core/db_connection.py` provides an in-memory SQLite DB that loads CSV/XLSX files found in the project root (e.g., `ventes_cleann.csv`, `STOCK.xlsx`) into tables named `t_<sanitized_filename>`.
  - `execute_select(sql, params, max_rows)` runs SQL and returns `columns`, `rows`, `rowcount`.
  - Handles common CSV encoding/delimiter issues and date normalization.
- `core/schema_loader.py` and `core/table_schemas.py` (if present) provide schema snapshots used by the QueryAgent for identifier checks and to present the schema to the LLM.

### `cache/`

- `cache/forecast_cache.py` describes a `ForecastCacheManager` (single-user optimized) with:
  - Per-article keys (no session in keys), data fingerprint validation in metadata
  - Summary forecast keys
  - TTL and model-version invalidation
  - Cache warming helpers and stats
- `cache/README.md` documents the production-grade cache design in detail (per-repo doc).

### `llm/`

- `llm/client.py` is a lightweight Groq adapter that handles rate-limiting and cooldowns. If `GROQ_API_KEY` is not set, it returns a helpful fallback response.
- The project uses adapters like `llm.sql_agent` to centralize prompt construction and parsing of model outputs.

---

## API endpoints (summary)

Routers are registered under the `/api` prefix. The primary routers are:

- `/api/sql-chat` - Text-to-SQL endpoints (generate SQL, schema listing, run queries). Uses QueryAgent and ExecutorAgent.
- `/api/forecasts` - Upload data, compute forecasts, fetch per-article or summary forecasts, manage cache.
- `/api/dashboard` - Dashboard metrics and aggregate endpoints.

Health and metadata endpoints (in `main.py`):

- `GET /health` - Basic health message and components status
- `GET /` - Root with links to docs and features

See router docstrings and function definitions under `api/routers` for specifics and parameter details.

---

## Data formats & expectations

The Streamlit dashboard and `SalesForecaster` expect datasets with at least these columns (but the code is flexible and does fuzzy normalization):

- `Ref Article` (string) — unique article reference / id
- `Année` or `Date` — either a year (for yearly forecasts) or a date column (for monthly forecasts)
- `CA HT NET` — sales numeric column (net sales)

Optional helpful columns include `Designation`, `Marque`, `Famille`, `Sous Famille`.

`SalesForecaster` will attempt to normalize columns using unicode normalization and fuzzy matching; errors provide descriptive messages listing available columns and suggestions.

---

## Testing

- Run unit tests with pytest if available:

  ```powershell
  pytest -q
  ```

- The repository contains tests under `tests/`; recommended tests include checks for the QueryAgent pipeline, SalesForecaster metrics, and cache behavior.

---

## Troubleshooting & notes

- If LLM responses are not produced, check `GROQ_API_KEY` and network access.
- For missing optional forecast backends:
  - Install `statsmodels` for ARIMA
  - Install `prophet` for Prophet forecasts
  - Install `xgboost` for XGBoost model
- If the in-memory SQLite cannot load CSV due to encoding issues, `core/db_connection` tries several encodings. Ensure file is accessible and has expected structure.
- Cache not found: check `cache/` folder permissions and whether `SalesForecaster` used an inferred `source_hash` that isolates caches per upload.

---

## Docker (quickstart)

This repository includes a `Dockerfile` and a `docker-compose.yml` to run the FastAPI app and the Streamlit dashboard in containers.

Files added:
- `Dockerfile` - builds a Python image with the project's dependencies and default command to run the FastAPI app.
- `docker-compose.yml` - defines two services:
  - `api` (FastAPI) mapped to port 8000
  - `streamlit` (Streamlit dashboard) mapped to port 8501
- `.env.example` - example env file you should copy to `.env` and customize (sensitive keys like `GROQ_API_KEY` should go into `.env`).

Important: docker-compose is configured to read environment variables from `.env` using `env_file: .env`. Make sure to copy `.env.example` to `.env` and set any secrets there. The `.env` file is included in `.dockerignore` by default so it will not be sent into the image build context or committed accidentally.

Quick run (PowerShell):

```powershell
# copy example env and edit values
copy .env.example .env
# build and run services (detached)
docker compose up --build -d

# view FastAPI docs at http://localhost:8000/docs
# view Streamlit at http://localhost:8501

# to follow logs for api service
docker compose logs -f api

# stop and remove containers
docker compose down
```

If you want only the FastAPI service (no streamlit), run:

```powershell
docker compose up --build api
```

Notes:
- The compose file mounts the repository into `/app` in the containers to allow live code edits during development. For production use, remove the volume mount and consider building a smaller image (e.g., with wheels and without build tools).
- If Docker is not available on Windows, use WSL2 or Docker Desktop.

