try:
    # pydantic v1 or v2 compatibility: BaseSettings moved to pydantic_settings in v2
    from pydantic import Field
    try:
        from pydantic import BaseSettings
    except Exception:
        # pydantic v2
        from pydantic_settings import BaseSettings
except Exception:
    # As a final fallback, provide a tiny BaseSettings-like shim
    from dataclasses import dataclass
    from typing import Any

    def Field(default: Any, **kwargs):
        return default

    @dataclass
    class BaseSettings:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

from typing import Optional


class Settings(BaseSettings):
    """Configuration loaded from environment or .env file.

    Fields are intentionally small and explicit so the rest of the code
    can import a single Settings instance.
    """

    MODEL_PATH: str = Field("Chat2DB/Chat2DB-SQL-7B", description="HuggingFace model id or local path")
    MODEL_USE_MOCK: bool = Field(False, description="When true, use a lightweight mock model (useful for tests)")
    MODEL_MAX_NEW_TOKENS: int = 256

    # DB URL examples:
    # sqlite: sqlite:///./data.db
    # postgres: postgresql+psycopg2://user:pass@localhost:5432/dbname
    # mysql: mysql+pymysql://user:pass@localhost:3306/dbname
    DATABASE_URL: Optional[str] = Field("sqlite:///:memory:", description="SQLAlchemy DB URL")

    BUSINESS_RULES_MD: str = Field("CHATBOT_ARCHITECTURE.md", description="Path to .md containing business rules (optional)")

    # Model generation defaults
    GENERATION_TEMPERATURE: float = 0.0
    GENERATION_TOP_P: float = 1.0

    class Config:
        env_file = ".env"
        extra = "ignore"


# pydantic v2 uses `model_config`; setting it here helps pydantic v2 accept extra env vars
try:
    Settings.model_config = {"extra": "ignore", "env_file": ".env"}
except Exception:
    pass

settings = Settings()
