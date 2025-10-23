"""
Core Configuration - Unified settings for the entire application

This module consolidates all configuration settings:
- Application settings (server, CORS, logging)
- LLM settings (Groq API with LangChain)
- Guardrails AI settings (SQL validation, PII protection)
- Data paths
- Agent settings

All settings can be overridden via environment variables.
"""

from pathlib import Path
import os
from typing import List
from dotenv import load_dotenv
import logging

# Load .env from project root if present
PROJECT_ROOT = Path(__file__).resolve().parents[1]
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)

logger = logging.getLogger("core.config")

# ============================================================================
# SERVER SETTINGS
# ============================================================================

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# CORS settings
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
ALLOWED_METHODS = ["*"]
ALLOWED_HEADERS = ["*"]

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

# Groq API key
GROQ_API_KEY = os.getenv("Groq_Api_key") or os.getenv("GROQ_API_KEY")

# Groq model name
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# LLM token limits
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 1024))
LLM_CONTEXT_TOP_K = int(os.getenv("LLM_CONTEXT_TOP_K", 6))

# Groq rate limit cooldown (hours)
GROQ_COOLDOWN_HOURS = int(os.getenv("GROQ_COOLDOWN_HOURS", 1))

# ============================================================================
# GUARDRAILS AI CONFIGURATION
# ============================================================================

# Enable PII masking in query results
ENABLE_PII_MASKING = os.getenv("ENABLE_PII_MASKING", "true").lower() in ("1", "true", "yes")

# SQL validation settings
SQL_MAX_QUERY_LENGTH = int(os.getenv("SQL_MAX_QUERY_LENGTH", "5000"))
SQL_ENFORCE_LIMIT = int(os.getenv("SQL_ENFORCE_LIMIT", "1000"))

# Guardrails logging
GUARDRAILS_LOG_VIOLATIONS = os.getenv("GUARDRAILS_LOG_VIOLATIONS", "true").lower() in ("1", "true", "yes")

# ============================================================================
# DATA PATHS
# ============================================================================

# Cache directories
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Forecast data
DATA_PATH = CACHE_DIR / "forecasts"
DATA_PATH.mkdir(parents=True, exist_ok=True)

# Logs
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Business data files
SALES_DATA_PATH = PROJECT_ROOT / "ventes_cleann.csv"
STOCK_DATA_PATH = PROJECT_ROOT / "STOCK.xlsx"

# ============================================================================
# FILE UPLOAD SETTINGS
# ============================================================================

MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
SUPPORTED_FILE_TYPES = [".csv", ".xlsx", ".xls"]

# ============================================================================
# AGENT SETTINGS
# ============================================================================

# Agent timeout (seconds)
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", 30))

# ============================================================================
# FORECASTING SETTINGS
# ============================================================================

DEFAULT_FORECAST_PERIOD = 3
DEFAULT_ALPHA = 0.3
DEFAULT_FAST_MODE = True

# ============================================================================
# DEPRECATED (Kept for backward compatibility)
# ============================================================================

# Legacy settings - no longer used
ENABLE_DATA_BACKED_ANSWERS = False
FORCE_REPO_SUMMARY = os.getenv("RAG_FORCE_REPO_SUMMARY", "0") in ("1", "true", "True")
