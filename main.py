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
