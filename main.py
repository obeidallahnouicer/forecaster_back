"""main.py

Orchestrator example demonstrating how to build embeddings from business rules
markdown and generate a safe SQL query using RAG + LLM.

This script is intended as an example and simple test harness. It:
 - Locates the business rules markdown file (tries a few common names).
 - Builds embeddings and an in-memory retrieval index.
 - Runs an example query and prints the SQL produced by the LLM.
 - Validates the SQL with `sql_validator`.

Before running:
 - Install dependencies: sentence-transformers (optional) and openai (if using OpenAI).
 - If using OpenAI, set environment variable OPENAI_API_KEY.
"""
from __future__ import annotations

import os
import sys
from typing import List

from tools.rag.embed_rules import build_embeddings_from_file
from tools.rag.rag_sql_generator import RAGSQLGenerator
from tools.rag.rag_sql_validator import enforce_sql_format


def find_rules_file() -> str:
    """Try to find the business rules markdown file in the repo root.

    Looks for `TABLE_Chatbot.md` or `TABLE Chatbot.md` (the project contains
    `TABLE Chatbot.md` with a space). If not found, raise FileNotFoundError.
    """
    candidates = ["TABLE_Chatbot.md", "TABLE Chatbot.md", "TABLE Chatbot.MD", "TABLE_Chatbot.MD"]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("Could not find business rules markdown file (looked for TABLE_Chatbot.md or TABLE Chatbot.md)")


def run_example(query: str) -> None:
    """Run the RAG SQL generator pipeline for an example query and print results.

    Args:
        query: Natural language query to convert to SQL.
    """
    rules_path = find_rules_file()
    print(f"Loading business rules from {rules_path}...\n")
    chunks, embeddings = build_embeddings_from_file(rules_path)

    print(f"Built {len(chunks)} chunks and embeddings. Initializing RAG generator...\n")
    rag = RAGSQLGenerator(chunks, embeddings)

    print(f"Generating SQL for query: {query}\n")
    sql = rag.generate_sql(query)
    # rag.generate_sql may return (sql, params) tuple or plain string
    if isinstance(sql, tuple) and len(sql) == 2:
        sql_text, sql_params = sql
    else:
        sql_text, sql_params = (sql or "", {})

    ok, normalized = enforce_sql_format(sql_text)
    if not ok:
        print("Generated SQL failed safety checks. Output was:\n")
        print(sql)
        sys.exit(2)

    # show SQL (comments allowed) and a short message
    print("--- GENERATED SQL (validated) ---\n")
    print(normalized)
    print("\n--- End SQL ---\n")


if __name__ == "__main__":
    # Example queries. Replace with other NL queries as needed.
    examples: List[str] = [
        "Quels clients sont inactifs depuis 6 mois ?",
        "Quels produits ont un stock supérieur à 6 mois ?",
    ]

    # Run first example by default
    run_example(examples[0])
"""
Sales Forecaster & Business Intelligence API - LangChain + Guardrails Edition

Unified FastAPI application providing:
- **Text-to-SQL chatbot** - Natural language to SQL with LangChain chains
- SQL safety with Guardrails AI (injection prevention, PII masking)
- Sales forecasting with multiple ML models
- Dashboard metrics and analytics

Architecture:
- /api/sql-chat - Text-to-SQL with LangChain + Guardrails AI
- /api/forecasts - Forecast upload and retrieval  
- /api/dashboard - Dashboard metrics

Powered by:
- LangChain for workflow orchestration
- Groq (LLaMA 3.3 70B) for LLM inference
- Guardrails AI for safety (SQL validation, injection prevention, PII masking)
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import logging

from core.config import (
    ALLOWED_ORIGINS, 
    ALLOWED_METHODS, 
    ALLOWED_HEADERS,
    LOG_LEVEL, 
    LOG_FORMAT, 
    DEBUG,
    HOST,
    PORT
)

# Import API routers
from api.routers import sql_chat, forecasts, dashboard


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper()),
        format=LOG_FORMAT
    )
    
    # Create FastAPI app
    app = FastAPI(
        title="Sales Forecaster & Business Intelligence API",
        description=(
            "Text-to-SQL chatbot with LangChain orchestration and Guardrails AI safety. "
            "Powered by LLaMA 3.3 70B (Groq) with PII masking and SQL injection prevention."
        ),
        version="4.0.0",
        debug=DEBUG,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=ALLOWED_METHODS,
        allow_headers=ALLOWED_HEADERS,
    )
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logger = logging.getLogger("api.requests")
        client = request.client.host if request.client else "unknown"
        logger.info(f"→ {request.method} {request.url.path} from {client}")
        
        try:
            response = await call_next(request)
            logger.info(f"← {request.method} {request.url.path} → {response.status_code}")
            return response
        except Exception as exc:
            logger.exception(f"✗ {request.method} {request.url.path}: {exc}")
            raise
    
    # Include API routers
    app.include_router(sql_chat.router, prefix="/api", tags=["SQL Chat"])
    app.include_router(forecasts.router, prefix="/api", tags=["Forecasts"])
    app.include_router(dashboard.router, prefix="/api", tags=["Dashboard"])
    
    # Health check endpoint
    @app.get("/health", tags=["System"])
    async def health_check():
        """System health check."""
        return {
            "status": "healthy",
            "version": "4.0.0",
            "api": {
                "sql_chat": "active",
                "forecasts": "active",
                "dashboard": "active"
            },
            "llm": "groq-llama-3.3-70b",
            "framework": "langchain",
            "safety": "guardrails-ai"
        }
    
    @app.get("/", tags=["System"])
    async def root():
        """API root - redirect to docs."""
        return {
            "message": "Sales Forecaster & Business Intelligence API - LangChain + Guardrails Edition",
            "version": "4.0.0",
            "docs": "/docs",
            "health": "/health",
            "endpoints": {
                "sql_chat": "/api/sql-chat",
                "schema": "/api/sql-chat/schema",
                "forecasts": "/api/forecasts",
                "dashboard": "/api/dashboard"
            },
            "features": [
                "LangChain workflow orchestration",
                "Guardrails AI SQL validation",
                "PII detection and masking",
                "SQL injection prevention"
            ]
        }
    
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    logger = logging.getLogger("main")
    logger.info(f"Starting server on {HOST}:{PORT}")
    logger.info(f"Debug mode: {DEBUG}")
    logger.info(f"Documentation: http://{HOST}:{PORT}/docs")
    
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL.lower()
    )
