# Codebase Index and Architecture Documentation

**Generated**: November 2, 2025  
**Repository**: forecaster_back  
**Branch**: dev  
**Primary Focus**: Chat2DB Chatbot Logic

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Core Architecture](#core-architecture)
3. [Module Index](#module-index)
4. [Chat2DB Integration](#chat2db-integration)
5. [Chatbot Flow](#chatbot-flow)
6. [Dependencies](#dependencies)

---

## Project Overview

This is a sales forecasting and business intelligence chatbot system that uses:
- **Chat2DB/Chat2DB-SQL-7B** as the primary SQL generation model
- Multi-agent architecture for query processing, execution, and insight generation
- SQLite database with sales, client, product, and stock data
- Business rules engine for automated recommendations

### Key Features
- Natural language to SQL conversion using Chat2DB
- Strict validation pipeline (PII detection, SQL validation)
- Business rules-based insights and recommendations
- Multi-layer caching system
- RESTful API with FastAPI
- Streamlit web interface

---

## Core Architecture

```
User Question (Natural Language)
    ↓
[QueryAgent] - Validates input, generates SQL via Chat2DB
    ↓
[ExecutorAgent] - Executes validated SQL queries
    ↓
[InsightAgent] - Generates business insights and recommendations
    ↓
Response (SQL results + insights + recommendations)
```

### Design Patterns
- **Agent Pattern**: Specialized agents for different concerns (query, execution, insight)
- **Validator Pattern**: Layered validation (PII, SQL structure, schema validation)
- **Adapter Pattern**: `llm/sql_agent.py` adapts multiple LLM backends
- **Strategy Pattern**: Multiple SQL generation strategies (Chat2DB, RAG, prompt-based)

---

## Module Index

### 1. **agents/** - Multi-Agent System

#### **query_agent.py** (849 lines)
**Purpose**: Orchestrates SQL generation with strict validation pipeline

**Key Classes**:
- `QueryAgent`: Main agent for text-to-SQL conversion

**Key Methods**:
- `generate_sql(question: str) -> Dict[str, Any]`: Full validation pipeline
  - Stage 1: PII detection
  - Stage 2: LLM generation (delegates to `llm/sql_agent.py`)
  - Stage 3: SQL output validation
  - Stage 4: Schema consistency checking
  - Auto-correction with fuzzy matching
  - LLM-driven retry mechanism

**Helper Functions**:
- `_extract_identifiers_from_sql(sql: str) -> set`: Extracts table/column names
- `_parameterize_intervals()`: Converts INTERVAL literals to parameters
- `_clean_sql()`: Normalizes SQL text
- `_format_schema()`: Schema formatting for prompts

**Dependencies**:
- `core.schema_loader`
- `guardrails.pii_detector`
- `guardrails.sql_validator`
- `llm.sql_agent` (primary SQL generation)
- `text2sql.agent` (Chat2DB backend)

---

#### **executor_agent.py** (80 lines)
**Purpose**: Executes validated SQL queries safely

**Key Classes**:
- `ExecutorAgent`: Query execution with safety checks

**Key Methods**:
- `execute(sql_query, validation_passed, params) -> Dict`: Executes if validated
- `execute_safe(sql_query) -> Dict`: Bypasses validation (internal use)

**Helper Functions**:
- `execute_query(sql, params, max_rows) -> Dict`: Standalone execution wrapper

**Safety Features**:
- Requires `validation_passed=True` flag
- Execution time tracking
- Row count limits
- Structured error handling

---

#### **insight_agent.py** (317 lines)
**Purpose**: Data-driven analysis using Business Rules Engine

**Key Classes**:
- `InsightAgent`: Generates business insights and recommendations

**Key Methods**:
- `analyze(rows, question, sql) -> Dict`: Full analysis pipeline
- `_compute_stats()`: Statistical analysis
- `_detect_numeric_columns()`: Auto-detect metrics
- `_generate_insights()`: Rule-based insights
- `_generate_recommendations()`: Actionable recommendations

**Integration**:
- Uses `core.business_rules.BusinessRulesEngine`
- LangChain + Groq LLM for natural language insights
- Entity-type detection (client, product, sales, stock)

---

#### **sql_validator.py**
**Purpose**: Legacy SQL validation (deprecated in favor of `guardrails/`)

**Status**: Redundant with `guardrails.sql_validator`

---

### 2. **text2sql/** - Chat2DB Integration Layer

#### **agent.py** (100+ lines)
**Purpose**: QueryAgent for Chat2DB backend

**Key Classes**:
- `QueryAgent`: Text-to-SQL with Chat2DB model

**Key Methods**:
- `generate_and_run(question, retries=3) -> Dict`: Full SQL generation flow
  - Schema loading
  - Prompt building with strict instructions
  - SQL extraction from `<SQL>...</SQL>` markers
  - Validation and auto-correction
  - Retry logic with exponential backoff
  - Query execution

**Workflow**:
1. Load DB schema
2. Build prompt with schema + question
3. Call Chat2DB model
4. Extract SQL between markers
5. Validate and auto-correct
6. Retry if validation fails
7. Execute and return results

---

#### **model_loader.py** (126 lines)
**Purpose**: Lazy loading of Chat2DB model or mock

**Key Classes**:
- `MockModel`: Deterministic mock for testing
  - Handles common questions from TABLE Chatbot.md
  - Returns predictable SQL
- `ModelLoader`: Loads real Chat2DB or mock

**Key Methods**:
- `load()`: Loads Chat2DB-SQL-7B or mock based on config
- `generate_sql(prompt) -> str`: Unified interface

**Model Details**:
- Default: `Chat2DB/Chat2DB-SQL-7B` from HuggingFace
- GPU acceleration if available
- Fallback to CPU
- Uses transformers pipeline

---

#### **config.py** (70 lines)
**Purpose**: Configuration for text2sql module

**Key Settings**:
- `MODEL_PATH`: "Chat2DB/Chat2DB-SQL-7B"
- `MODEL_USE_MOCK`: False (set to True for tests)
- `DATABASE_URL`: SQLite/PostgreSQL/MySQL connection
- `GENERATION_TEMPERATURE`: 0.0 (deterministic)
- `GENERATION_TOP_P`: 1.0

**Pydantic v1/v2 Compatible**

---

#### **db.py** (50 lines)
**Purpose**: Database connection wrapper

**Key Classes**:
- `DBConnection`: SQLAlchemy-based DB access

**Key Methods**:
- `get_schema() -> Dict[str, List[str]]`: Introspect schema
- `execute_select(sql, params) -> List[Dict]`: Safe SELECT execution

---

#### **validator.py** (111 lines)
**Purpose**: SQL validation and auto-correction

**Key Functions**:
- `is_select_only(sql) -> bool`: Ensures only SELECT statements
- `extract_identifiers(sql) -> List[str]`: Extract table/column names
- `fuzzy_fix_identifier(name, choices) -> Tuple`: Fuzzy matching
- `validate_and_autocorrect(sql, schema) -> Tuple[str, List[str]]`: Main validator

**Features**:
- Rejects INSERT/UPDATE/DELETE/DROP/ALTER
- Uses rapidfuzz for fuzzy matching (optional)
- Auto-corrects misspelled identifiers
- Returns corrected SQL + issues list

---

#### **schema_inspector.py**
**Purpose**: Schema introspection helper

**Key Functions**:
- `load_schema(db_connection) -> Dict`: Loads full schema

---

#### **rule_engine.py**
**Purpose**: Business rules processing from markdown

**Key Functions**:
- Parses TABLE Chatbot.md
- Extracts business logic
- Provides rule-based corrections

---

### 3. **llm/** - LLM Client and Adapter Layer

#### **sql_agent.py** (217 lines) ⭐ **PRIMARY ADAPTER**
**Purpose**: Thin adapter that orchestrates SQL generation across multiple backends

**Key Functions**:
- `generate_sql_from_question(question, sample_rows, ...) -> Dict`: Main entry point
  - **Primary Path**: Delegates to `text2sql.agent.QueryAgent` (Chat2DB)
  - **Fallback Path**: Legacy prompt-based generation with `llm.client`
  - Returns normalized response: `{reasoning, sql_query, params, raw, meta}`

**Backend Priority**:
1. **text2sql QueryAgent** (Chat2DB-SQL-7B) - PREFERRED
2. Legacy prompt-based generation (fallback)
3. Optional RAG integration (if enabled)

**Helper Functions**:
- `parse_and_normalize_llm_response()`: Backwards-compatible response normalization
- `_load_or_build_embeddings()`: RAG embeddings cache
- `_find_rules_file()`: Locate TABLE Chatbot.md

**Response Schema**:
```python
{
    "reasoning": str,        # Explanation of SQL logic
    "sql_query": str,        # SELECT statement
    "params": dict,          # Query parameters
    "raw": str,              # Raw LLM output
    "meta": dict            # Debugging metadata
}
```

---

#### **client.py**
**Purpose**: Generic LLM client for non-SQL tasks

**Key Functions**:
- `get_llm_response(prompt, model, ...) -> Dict`: Generic LLM calls
- Used for insights, summaries, explanations

---

#### **token_manager.py**
**Purpose**: Token counting and rate limiting

---

### 4. **guardrails/** - Validation Layer

#### **pii_detector.py**
**Purpose**: Detect and block PII in user input

**Key Classes**:
- `PIIInputValidator`: Regex-based PII detection

**Detects**:
- Email addresses
- Phone numbers
- Credit card numbers
- Social security numbers

---

#### **sql_validator.py**
**Purpose**: SQL output validation

**Key Classes**:
- `SQLOutputValidator`: Validates generated SQL

**Checks**:
- SELECT-only enforcement
- SQL injection patterns
- Dangerous keywords
- Basic syntax validation

---

### 5. **core/** - Core Infrastructure

#### **config.py**
**Purpose**: Global configuration

**Key Settings**:
- `GROQ_API_KEY`: LLM API key
- `GROQ_MODEL`: Default model
- `DATABASE_PATH`: SQLite DB path
- `LLM_RETRY_ATTEMPTS`: Retry count

---

#### **db_connection.py**
**Purpose**: Database connection and query execution

**Key Functions**:
- `get_db_connection() -> sqlite3.Connection`: Get DB connection
- `execute_select(sql, params, max_rows) -> Dict`: Safe SELECT execution
- `get_schema() -> Dict`: Schema introspection

---

#### **schema_loader.py**
**Purpose**: Schema snapshot and formatting

**Key Functions**:
- `get_schema_snapshot(limit_sample) -> Dict`: Load schema with samples
- Caches schema for performance

---

#### **table_schemas.py**
**Purpose**: Enhanced schema with business rules

**Key Functions**:
- `format_schema_for_llm() -> str`: Rich schema for prompts
- Includes column descriptions, data types, business rules

---

#### **business_rules.py**
**Purpose**: Business Rules Engine

**Key Classes**:
- `BusinessRulesEngine`: Rule evaluation
- `EntityType`: Enum for entity types (CLIENT, PRODUCT, SALES, STOCK)

**Key Methods**:
- `evaluate_rules(entity_type, data) -> List[Dict]`: Apply rules
- `get_recommendations() -> List[str]`: Generate recommendations

**Rules Covered**:
- Client inactivity (> 3 months)
- Client churn risk (CA drop > 20%)
- Product overstock (coverage > 6 months)
- Product stockout risk (coverage < 1 month)
- Sales trends (growth/decline detection)

---

#### **context_manager.py**
**Purpose**: Conversation context management

**Key Functions**:
- Tracks conversation history
- Maintains session state
- Provides context for follow-up questions

---

#### **registry.py**
**Purpose**: Service registry pattern

---

### 6. **cache/** - Caching System

**Modules**:
- `manager.py`: Cache orchestration
- `simple_cache.py`: In-memory cache
- `forecast_cache.py`: Forecast results caching
- `upload_cache.py`: File upload caching
- `sessions.json`: Session persistence

**Cache Types**:
- SQL query results
- LLM responses
- Forecast computations
- File uploads
- RAG embeddings

---

### 7. **prompts/** - Prompt Templates

**Key Files**:
- `sql_generation.py`: SQL generation prompts
  - `SQL_SYSTEM_PROMPT`: System instructions
  - `SQL_GENERATION_TEMPLATE`: User prompt template
  - `build_sql_generation_prompt()`: Prompt builder

- `insight_generation.py`: Insight prompts
  - `INSIGHT_GENERATION_CHAT_PROMPT`: LangChain prompt
  - `generate_file_schema()`: Schema formatter

---

### 8. **tools/** - Utility Tools

**Subdirectories**:
- `rag/`: RAG (Retrieval-Augmented Generation)
  - `rag_sql_generator.py`: RAG-based SQL generation
  - `embed_rules.py`: Business rules embeddings

---

### 9. **api/** - REST API Layer

**Structure**:
- `routers/`: API endpoints
- `schemas.py`: Pydantic request/response models

**Framework**: FastAPI

---

### 10. **tests/** - Test Suite

**Test Coverage**:
- Unit tests for agents
- Integration tests for Chat2DB
- Mock model tests
- Validation tests

---

## Chat2DB Integration

### Architecture Overview

```
User Question
    ↓
agents/query_agent.py (orchestrator)
    ↓
llm/sql_agent.py (adapter)
    ↓
text2sql/agent.py (Chat2DB backend)
    ↓
text2sql/model_loader.py (Chat2DB-SQL-7B or MockModel)
    ↓
text2sql/validator.py (validation & auto-correction)
    ↓
text2sql/db.py (execution)
    ↓
SQL Results
```

### Key Integration Points

1. **Primary Entry**: `llm/sql_agent.generate_sql_from_question()`
2. **Chat2DB Call**: Lines 108-127 in `llm/sql_agent.py`
3. **Model Loading**: `text2sql/model_loader.ModelLoader`
4. **Prompt Format**: `<SQL>...</SQL>` marker extraction
5. **Validation**: `text2sql/validator.validate_and_autocorrect()`

### Configuration

**Environment Variables**:
- `MODEL_PATH`: "Chat2DB/Chat2DB-SQL-7B" (default)
- `MODEL_USE_MOCK`: "0" (use real model) or "1" (use mock)
- `ENABLE_RAG`: "1" (enable RAG) or "0" (disable)
- `DATABASE_URL`: SQLite/PostgreSQL/MySQL connection string

### Mock Model Behavior

The `MockModel` in `text2sql/model_loader.py` provides deterministic responses for testing:

**Supported Questions**:
- "Combien de clients?" → `SELECT COUNT(*) FROM clients`
- "Clients inactifs" → `SELECT Code_Client, Intitule_Client, Mois_Depuis_Derniere_Vente FROM clients WHERE Mois_Depuis_Derniere_Vente >= 3`
- "CA moyen client X" → `SELECT CA_Moyen_Annuel FROM clients WHERE Intitule_Client = 'X'`
- "Risque de perte" → `SELECT Code_Client, Intitule_Client, Variation_CA_Percent FROM clients WHERE Variation_CA_Percent <= -20`

---

## Chatbot Flow

### End-to-End Request Flow

1. **User Input** → Streamlit/API receives question
2. **QueryAgent.generate_sql()**:
   - PII validation
   - Call `llm/sql_agent.generate_sql_from_question()`
   - SQL validation
   - Schema consistency check
   - Auto-correction if needed
3. **ExecutorAgent.execute()**:
   - Execute validated SQL
   - Return structured results
4. **InsightAgent.analyze()**:
   - Compute statistics
   - Apply business rules
   - Generate insights
   - Provide recommendations
5. **Response** → JSON with SQL, data, insights, recommendations

### Validation Pipeline

```
Input Question
    ↓
[PII Detector] - Block sensitive data
    ↓
[LLM Generation] - Chat2DB produces SQL
    ↓
[SQL Validator] - Structure validation
    ↓
[Schema Checker] - Identifier validation
    ↓
[Auto-Corrector] - Fuzzy matching
    ↓
[LLM Retry] - Regenerate if needed
    ↓
Validated SQL
```

### Error Handling

**Validation Stages**:
- `input_validation`: PII detected
- `llm_generation`: Model failed to generate SQL
- `output_validation`: SQL structure invalid
- `schema_validation`: Unknown tables/columns

**Retry Strategy**:
- Max retries: 2 (configurable)
- Exponential backoff
- LLM-driven correction with schema context

---

## Dependencies

### Core Dependencies
- **transformers**: HuggingFace model loading
- **torch**: PyTorch for Chat2DB model
- **sqlalchemy**: Database abstraction
- **pydantic**: Configuration and validation
- **fastapi**: REST API framework
- **streamlit**: Web UI
- **langchain**: LLM orchestration
- **langchain-groq**: Groq LLM provider

### Optional Dependencies
- **rapidfuzz**: Fuzzy string matching
- **sqlparse**: SQL parsing
- **numpy**: Embeddings cache

### Development Dependencies
- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Linting

---

## Use Cases from TABLE Chatbot.md

### Client Queries
1. ✅ "Quels sont les clients inactifs depuis plus de 3 mois?"
2. ✅ "Quel est le chiffre d'affaires moyen du client AYADI HOUWAIDA?"
3. ✅ "Quels clients ont augmenté leurs achats de plus de 20%?"
4. ✅ "Quels clients gère le commercial RABIAA BEN SEDRINE?"
5. ✅ "Quels sont les clients à risque de perte?"
6. ✅ "Top 10 des clients les plus fidèles?"

### Product Queries
1. ✅ "À quelle famille appartient l'article 32014?"
2. ✅ "Quel est le produit le plus vendu de la marque H.ZONE?"
3. ✅ "Est-ce que la famille 'Shampooing' est en croissance?"
4. ✅ "Quels produits ont un stock supérieur à 6 mois?"
5. ✅ "Quels produits risquent la rupture le mois prochain?"

### Sales Queries
1. ✅ "Comment évoluent les ventes de septembre à octobre?"
2. ✅ "Quel produit a généré le plus de CA ce mois-ci?"
3. ✅ "Quels clients ont baissé leur CA en octobre?"
4. ✅ "Compare les ventes de septembre 2025 et septembre 2024"

### Stock Queries
1. ✅ "Quels produits sont en surstock?"
2. ✅ "Quels produits risquent la rupture?"
3. ✅ "Quelle est la couverture de stock moyenne?"

---

## Notes

- **Primary Model**: Chat2DB-SQL-7B (7 billion parameters)
- **Database**: SQLite (local), supports PostgreSQL/MySQL
- **Language**: French (primary), English (supported)
- **Deployment**: Docker + docker-compose
- **API**: REST (FastAPI)
- **UI**: Streamlit web app

---

## Refactoring Recommendations

### High Priority
1. ✅ Consolidate duplicate `QueryAgent` classes (agents/ vs text2sql/)
2. ⚠️ Remove redundant `sql_validator.py` in agents/
3. ⚠️ Standardize response schemas across modules
4. ⚠️ Extract common validation logic into shared utilities

### Medium Priority
5. ⚠️ Add comprehensive type hints
6. ⚠️ Improve error messages with user-friendly explanations
7. ⚠️ Add telemetry and monitoring hooks
8. ⚠️ Optimize caching strategies

### Low Priority
9. ⚠️ Add docstring standards compliance
10. ⚠️ Refactor long functions (>100 lines)
11. ⚠️ Extract magic numbers to constants
12. ⚠️ Add integration tests for full chatbot flow

---

**End of Index**
