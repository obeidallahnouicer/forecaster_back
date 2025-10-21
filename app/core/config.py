"""
Configuration settings for the FastAPI application.
"""

from pathlib import Path
import os
from typing import List

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# CORS settings
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
ALLOWED_METHODS = ["*"]
ALLOWED_HEADERS = ["*"]

# File upload settings
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
UPLOAD_DIR = PROJECT_ROOT / "tmp_uploads"
SERVER_DATA_DIR = PROJECT_ROOT / "server_data"
CACHE_DIR = PROJECT_ROOT / "cache"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SERVER_DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Supported file types
SUPPORTED_FILE_TYPES = [".csv", ".xlsx", ".xls"]

# Forecasting settings
DEFAULT_FORECAST_PERIOD = 3
DEFAULT_ALPHA = 0.3
DEFAULT_FAST_MODE = True

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"
