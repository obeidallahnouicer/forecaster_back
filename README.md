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

│  Checks:  Email, phone, SSN, credit card, sensitive keywords   │

└────────────────────────────────────────────────────────────────┘# Run server

                              ↓python main.py

┌────────────────────────────────────────────────────────────────┐```

│  STAGE 2: LLM GENERATION (LangChain + Groq)                    │

│  ────────────────────────────────────────────────────────────  │Server starts at `http://localhost:8000`

│  Purpose: Generate SQL from natural language                   │- API Docs: `http://localhost:8000/docs`

│  Action:  ✓ SUCCESS → Continue to validation                   │- Health: `http://localhost:8000/health`

│           ✗ FAILURE → REJECT (return error to user)            │

│  Model:   Groq Llama 3.3 70B (via LangChain)                   │---

└────────────────────────────────────────────────────────────────┘

                              ↓## 📡 API Endpoints

┌────────────────────────────────────────────────────────────────┐

│  STAGE 3: OUTPUT VALIDATION (SQL Validator)                    │### 🤖 Text-to-SQL Chatbot

│  ────────────────────────────────────────────────────────────  │

│  Purpose: Validate generated SQL for safety and correctness    │#### **POST /api/sql-chat**

│  Action:  ✓ PASS → Continue to execution                       │Ask natural language questions about your data.

│           ✗ FAIL → REJECT (do NOT execute SQL, return error)   │

│  Checks:  Syntax, SQL injection, forbidden operations, logic   │**Request:**

└────────────────────────────────────────────────────────────────┘```json

                              ↓{

┌────────────────────────────────────────────────────────────────┐  "question": "What are the top 5 products by total sales?",

│  STAGE 4: EXECUTION (Executor Agent)                           │  "session_id": "user_123",

│  ────────────────────────────────────────────────────────────  │  "enforce_limit": true

│  Purpose: Execute validated SQL (read-only)                    │}

│  Action:  ✓ SUCCESS → Continue to insights                     │```

│           ✗ FAILURE → REJECT (return execution error)          │

│  Safety:  Only executes pre-validated queries                  │**Response:**

└────────────────────────────────────────────────────────────────┘```json

                              ↓{

┌────────────────────────────────────────────────────────────────┐  "success": true,

│  STAGE 5: INSIGHTS (Insight Agent)                             │  "question": "What are the top 5 products by total sales?",

│  ────────────────────────────────────────────────────────────  │  "sql": "SELECT CODE_PRODUIT, SUM(CA) as total_ca FROM t_ventes_cleann GROUP BY CODE_PRODUIT ORDER BY total_ca DESC LIMIT 5",

│  Purpose: Analyze results and generate recommendations         │  "params": [],

│  Action:  Return summary, insights, recommendations            │  "columns": ["CODE_PRODUIT", "total_ca"],

│  Method:  LangChain chain with analysis prompt                 │  "rows": [...],

└────────────────────────────────────────────────────────────────┘  "insights": {

```    "total_rows": 5,

    "numeric_summary": {...},

### Safety Guarantees    "key_insights": ["Top product generates 45% of total revenue"],

    "recommendations": ["Focus on top 3 products for growth"]

✅ **No PII sent to LLM** - Blocked at input validation    },

✅ **No SQL injection** - All SQL validated before execution    "confidence": 0.95,

✅ **Only SELECT queries** - Forbidden operations (DROP, DELETE, etc.) blocked    "metadata": {...}

✅ **No invalid SQL executed** - Syntax and structure validated  }

✅ **Fail-fast behavior** - Any validation failure stops workflow immediately```



------



## 🚀 Quick Start## 🏗️ Architecture



### Prerequisites### Clean Folder Structure



- Python 3.9+```

- Groq API key ([get one here](https://console.groq.com/))forecaster_back/

├── agents/                    # LangChain-powered agents

### Installation│   ├── query_agent.py        # NL → SQL using LangChain

│   ├── sql_validator.py      # Guardrails SQL validation

```bash│   ├── executor_agent.py     # Query execution + PII masking

# Clone repository│   └── insight_agent.py      # LangChain insights generation

git clone <repo-url>├── guardrails/                # Guardrails AI safety modules

cd forecaster_back│   ├── sql_validator.py      # Comprehensive SQL validation

│   ├── injection_detector.py # SQL injection prevention

# Install dependencies│   └── pii_detector.py       # PII detection and masking

pip install -r requirements.txt├── prompts/                   # Modular LangChain prompt templates

│   ├── sql_generation.py     # SQL generation prompts

# Set environment variables│   ├── insight_generation.py # Insight analysis prompts

export GROQ_API_KEY=your_api_key_here│   └── validation.py         # Validation prompts

├── api/

# Run server│   └── routers/

python main.py│       ├── sql_chat.py       # Text-to-SQL endpoint

```│       ├── forecasts.py      # Forecast endpoints

│       └── dashboard.py      # Dashboard metrics

Server starts at `http://localhost:8000`├── core/

│   ├── config.py             # Configuration (LangChain + Guardrails)

**Important URLs**:│   ├── db_connection.py      # In-memory SQLite loader

- API Documentation: `http://localhost:8000/docs`│   ├── schema_loader.py      # DB schema introspection

- Health Check: `http://localhost:8000/health`│   └── registry.py           # Forecast session registry

- Validation Info: `http://localhost:8000/api/sql-chat/validation-info`├── llm/

│   ├── client.py             # Groq LLM wrapper (legacy)

---│   └── token_manager.py      # Token tracking

├── tests/

## 📡 API Endpoints│   └── test_sql_chat.py      # Integration tests

├── main.py                    # FastAPI application

### 🤖 SQL Chatbot└── requirements.txt           # Dependencies (includes LangChain)

```

#### **POST /api/sql-chat**

### Text-to-SQL Workflow

Ask natural language questions about your data with strict validation.

```

**Request:**User Question

```json    ↓

{[Query Agent - LangChain] → Generate SQL with structured output

  "question": "Show me total sales by month in 2024",    ↓

  "session_id": "user_123"  // optional[SQL Validator - Guardrails] → Validate syntax, prevent injection

}    ↓

```[Injection Detector - Guardrails] → Advanced injection detection

    ↓

**Response (Success):**[Executor Agent] → Execute query + mask PII

```json    ↓

{[PII Detector - Guardrails] → Detect and mask sensitive data

  "success": true,    ↓

  "question": "Show me total sales by month in 2024",[Insight Agent - LangChain] → Analyze results + generate insights

  "answer": "Analysis of monthly sales for 2024 shows...",    ↓

  "sql": "SELECT strftime('%Y-%m', Date) as month, SUM(Montant) as total FROM t_base WHERE strftime('%Y', Date) = '2024' GROUP BY month",JSON Response (SQL + masked data + insights)

  "insights": [```

    "12 months of data analyzed",

    "Total sales: $1.2M",---

    "Peak month: December ($150K)"

  ],## 🧪 Testing

  "recommendations": [

    "Focus marketing efforts on Q4",```bash

    "Investigate Q1 dip"# Run all tests

  ],pytest

  "rows_preview": [

    {"month": "2024-01", "total": 95000},# Run SQL chat tests only

    {"month": "2024-02", "total": 87000}pytest tests/test_sql_chat.py -v

  ],

  "rowcount": 12,# Test with coverage

  "execution_time_ms": 245.3,pytest --cov=agents --cov=api --cov=core

  "validation_status": {```

    "input_validation": "passed",

    "llm_generation": "passed",**Current Test Status**: ✅ All 5 tests passing

    "output_validation": "passed",

    "execution": "passed",---

    "insights": "passed"

  }## 🎯 Design Principles

}

```1. **LangChain First**: All LLM interactions use LangChain chains for maintainability

2. **Safety by Default**: Guardrails AI validates every SQL query before execution

**Response (Validation Failure):**3. **PII Protection**: Automatic detection and masking of sensitive data

```json4. **Data-Driven**: All insights come from actual data, not hallucinations

{5. **SQL-First**: Direct SQL queries on structured data (no RAG overhead)

  "success": false,6. **Modular Prompts**: Reusable prompt templates in dedicated folder

  "question": "Show me john.doe@email.com purchases",7. **Testable**: Clear separation of concerns for easy testing

  "error": "INPUT REJECTED: PII or sensitive data detected (email). Please rephrase your query without including personal information.",

  "execution_time_ms": 12.5,---

  "validation_status": {

    "input_validation": "failed",## 📦 Data Sources

    "llm_generation": "not_started",

    "output_validation": "not_started",The system auto-loads CSV/Excel files from the root directory into in-memory SQLite:

    "execution": "not_started",

    "insights": "not_started"- `BASE.xlsx` → `t_base`

  }- `STOCK.xlsx` → `t_stock`

}- `ventes_cleann.csv` → `t_ventes_cleann`

```- `forecast-summary.csv` → `t_forecast_summary`



#### **GET /api/sql-chat/health**Schema is dynamically introspected and provided to the LLM for SQL generation.



Check service health and configuration.---



**Response:**## 🔑 Environment Variables

```json

{```bash

  "status": "healthy",# Required

  "service": "sql-chat",GROQ_API_KEY=your_groq_api_key

  "pipeline": "LangChain + Guardrails AI",

  "llm": "llama-3.3-70b-versatile",# Optional (with defaults)

  "validation": {HOST=0.0.0.0

    "input_validator": "PIIInputValidator",PORT=8000

    "output_validator": "SQLOutputValidator"LOG_LEVEL=INFO

  }DEBUG=False

}GROQ_MODEL=llama-3.3-70b-versatile

```GROQ_RPM=30



#### **GET /api/sql-chat/validation-info**# Guardrails AI Settings

ENABLE_PII_MASKING=true

Get detailed information about the validation pipeline.SQL_MAX_QUERY_LENGTH=5000

SQL_ENFORCE_LIMIT=1000

**Response:**GUARDRAILS_LOG_VIOLATIONS=true

```json```

{

  "pipeline": {---

    "1_input_validation": {

      "purpose": "Check for PII/sensitive data in user input",## 📝 Example Questions

      "action_on_fail": "REJECT - Do not proceed to LLM",

      "validator": "PIIInputValidator",**Sales Analysis:**

      "checks": ["email", "phone", "ssn", "credit_card", "sensitive_keywords"]- "What are the top 10 products by revenue?"

    },- "Show me total CA grouped by month"

    "2_llm_generation": {...},- "Which clients have purchased in the last 3 months?"

    "3_output_validation": {...},

    "4_execution": {...},**Stock Analysis:**

    "5_insights": {...}- "List products with stock below 100 units"

  },- "Show stock coverage by product category"

  "safety_guarantees": [

    "No PII/sensitive data sent to LLM",**Trends:**

    "No invalid SQL executed",- "What's the year-over-year revenue growth?"

    ...- "Show monthly sales trends for 2024"

  ]

}---

```

## 🔄 Migration from v3.x

---

**Removed:**

### 📈 Sales Forecasting- Direct LLM client calls (replaced with LangChain chains)

- Old RAG workflow (embeddings, ChromaDB, vectorstore)

#### **POST /api/forecasts/upload**- Hardcoded prompts (moved to `prompts/` folder)

- Basic SQL validation (upgraded to Guardrails AI)

Upload sales data for forecasting.

**Added:**

**Request:** `multipart/form-data` with CSV file- LangChain integration for all LLM interactions

- Guardrails AI for comprehensive safety:

**Response:**  - SQL syntax validation

```json  - SQL injection prevention

{  - PII detection and masking

  "success": true,- Modular prompt templates

  "message": "Data uploaded successfully",- Enhanced error handling and logging

  "rows": 1000,- PII masking in query results

  "columns": ["Date", "Product", "Sales"]

}**Upgrade Steps:**

```1. Install new dependencies: `pip install -r requirements.txt`

2. Set `GROQ_API_KEY` environment variable

#### **POST /api/forecasts/generate**3. Optional: Configure Guardrails settings (see Environment Variables)

4. Test with `/health` endpoint to verify setup

Generate forecast for uploaded data.

---

**Request:**

```json## 📞 Support

{

  "periods": 30,  // days to forecastFor issues or questions, check `/docs` or review `tests/test_sql_chat.py`.

  "model": "prophet"
}
```

**Response:**
```json
{
  "success": true,
  "forecast": [
    {"ds": "2024-11-01", "yhat": 15000, "yhat_lower": 13500, "yhat_upper": 16500},
    ...
  ],
  "metrics": {
    "mae": 250.5,
    "rmse": 320.8
  }
}
```

---

## 🏗️ Project Structure

```
forecaster_back/
├── agents/                      # LangChain agents
│   ├── query_agent.py          # SQL generation with validation
│   ├── executor_agent.py       # Validated SQL execution
│   └── insight_agent.py        # Result analysis with LangChain
│
├── guardrails/                  # Guardrails AI validators
│   ├── pii_detector.py         # INPUT validator (PII check)
│   └── sql_validator.py        # OUTPUT validator (SQL check)
│
├── prompts/                     # Modular prompt templates
│   ├── sql_generation.txt      # SQL generation prompt
│   ├── sql_validation.txt      # SQL validation prompt
│   └── insight_generation.txt  # Analysis prompt
│
├── api/
│   └── routers/
│       ├── sql_chat.py         # SQL chatbot endpoints
│       └── forecasts.py        # Forecasting endpoints
│
├── core/
│   ├── config.py               # Configuration
│   ├── db_connection.py        # Database connection
│   └── schema_loader.py        # Schema extraction
│
├── llm/
│   ├── client.py               # Groq LLM client
│   └── token_manager.py        # Token usage tracking
│
├── main.py                      # FastAPI application
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

---

## 🔧 Configuration

### Environment Variables

```bash
# Required
GROQ_API_KEY=your_groq_api_key

# Optional
GROQ_MODEL=llama-3.3-70b-versatile  # Default
DATABASE_PATH=./data/sales.db        # Default
LOG_LEVEL=INFO                       # Default
```

### Guardrails Configuration

Edit `core/config.py` to customize validation behavior:

```python
class Config:
    # PII Input Validation
    PII_STRICT_MODE = True              # Reject on PII detection
    
    # SQL Output Validation
    SQL_STRICT_MODE = True              # Reject on validation failure
    SQL_ALLOW_SUBQUERIES = True         # Allow subqueries in SQL
    
    # LangChain
    GROQ_MODEL = "llama-3.3-70b-versatile"
    GROQ_TEMPERATURE = 0.1              # Low temp for deterministic output
```

---

## 📚 Workflow Examples

### Example 1: Successful Query

```
User Input: "Show me top 5 customers by total sales"

1. INPUT VALIDATION:
   ✓ No PII detected → PASS

2. LLM GENERATION:
   Generated SQL: SELECT Code_Client, SUM(Montant) as total 
                  FROM t_base GROUP BY Code_Client 
                  ORDER BY total DESC LIMIT 5
   ✓ SQL generated → PASS

3. OUTPUT VALIDATION:
   ✓ Valid syntax → PASS
   ✓ No injection patterns → PASS
   ✓ Only SELECT → PASS
   ✓ Valid structure → PASS

4. EXECUTION:
   ✓ Query executed → 5 rows returned

5. INSIGHTS:
   Summary: "Top 5 customers account for 35% of total revenue..."
   Insights: ["Customer #123 is top performer", ...]
   Recommendations: ["Focus retention on top customers"]

Result: SUCCESS
```

### Example 2: PII Detected (Input Validation Failure)

```
User Input: "Show purchases for john.doe@company.com"

1. INPUT VALIDATION:
   ✗ PII detected (email) → FAIL

Result: REJECTED
Error: "INPUT REJECTED: PII or sensitive data detected (email). 
        Please rephrase your query without including personal information."

Pipeline stopped at Stage 1 - LLM never called.
```

### Example 3: SQL Injection Attempt (Output Validation Failure)

```
User Input: "Show all products; DROP TABLE t_base;"

1. INPUT VALIDATION:
   ✓ No PII → PASS

2. LLM GENERATION:
   (Hypothetically, LLM generates malicious SQL)
   Generated SQL: SELECT * FROM products; DROP TABLE t_base;
   ✓ SQL generated → PASS

3. OUTPUT VALIDATION:
   ✗ Multiple statements detected → FAIL
   ✗ Forbidden operation (DROP) → FAIL

Result: REJECTED
Error: "SQL validation FAILED: Forbidden operation 'DROP' - only SELECT queries allowed"

Pipeline stopped at Stage 3 - SQL never executed.
```

---

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run validation tests only
pytest tests/test_validation.py

# Run with coverage
pytest --cov=guardrails --cov=agents
```

### Manual Testing

```bash
# Test PII detection
curl -X POST http://localhost:8000/api/sql-chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Show me john@email.com orders"}'

# Test valid query
curl -X POST http://localhost:8000/api/sql-chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Show me total sales by month"}'

# Get validation info
curl http://localhost:8000/api/sql-chat/validation-info
```

---

## 🔒 Security Features

### Input Validation (PIIInputValidator)

**Detected PII Types**:
- Email addresses
- Phone numbers
- Social Security Numbers (SSN)
- Credit card numbers
- Passport numbers
- IBAN codes

**Sensitive Keywords**:
- password, secret, credential, token
- social security, credit card
- personal information, confidential

**Action**: Immediate rejection - no LLM call made

### Output Validation (SQLOutputValidator)

**Injection Patterns Blocked**:
- Multiple statements (`;`)
- Comment injection (`--`, `/**/`)
- Union-based injection (`UNION SELECT`)
- Boolean injection (`OR 1=1`)
- Stored procedure calls (`xp_`, `sp_`)

**Forbidden Operations**:
- DROP, TRUNCATE, DELETE
- INSERT, UPDATE, ALTER
- CREATE, GRANT, REVOKE
- EXEC, EXECUTE, SHUTDOWN

**Action**: Rejection before execution - SQL never runs

---

## 📊 Monitoring & Logging

### Logs

Logs are written to:
- `logs/agent_reasoning.log` - Agent decisions and reasoning
- `logs/agent_errors.log` - Errors and validation failures

### Validation Tracking

Every request returns `validation_status` showing which stages passed/failed:

```json
{
  "validation_status": {
    "input_validation": "passed",
    "llm_generation": "passed",
    "output_validation": "failed",  // ← Failure point
    "execution": "not_started",
    "insights": "not_started"
  }
}
```

---

## 🚧 Troubleshooting

### Common Issues

**Issue**: "INPUT REJECTED: PII detected"
- **Cause**: User input contains email, phone, or sensitive data
- **Fix**: Rephrase query without personal information

**Issue**: "SQL validation FAILED: Forbidden operation"
- **Cause**: LLM generated SQL with forbidden keywords (DROP, DELETE, etc.)
- **Fix**: Rephrase query to be read-only; check prompt templates

**Issue**: "LLM generation failed"
- **Cause**: Groq API error or invalid API key
- **Fix**: Check `GROQ_API_KEY` environment variable

**Issue**: "Query execution failed"
- **Cause**: SQL references non-existent tables/columns
- **Fix**: Check database schema with `/api/sql-chat/schema`

---

## 🤝 Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature/new-validation`
3. Implement changes with tests
4. Run tests: `pytest`
5. Submit pull request

**Important**: All validation logic changes must include tests proving:
- Unsafe input is rejected
- Safe input passes through
- No bypass is possible

---

## 📝 License

MIT License - See LICENSE file for details

---

## 📞 Support

- **Documentation**: See `/docs` endpoint
- **Validation Info**: GET `/api/sql-chat/validation-info`
- **Health Check**: GET `/api/sql-chat/health`

---

**Built with**: LangChain, Guardrails AI, Groq, FastAPI, SQLite  
**Version**: 5.0.0 - Strict Validation Pipeline  
**Last Updated**: October 2025
