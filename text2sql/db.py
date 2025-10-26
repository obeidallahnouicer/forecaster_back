from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from typing import Dict, Any, List
from .config import settings
from .logger import logger


class DBConnection:
    """Lightweight DB connection wrapper using SQLAlchemy.

    Provides safe execution for SELECT queries and utilities to inspect schema.
    """

    def __init__(self, db_url: str = None):
        self.db_url = db_url or settings.DATABASE_URL
        self._engine: Engine | None = None

    def get_engine(self) -> Engine:
        if self._engine is None:
            logger.info("Creating SQLAlchemy engine for %s", self.db_url)
            self._engine = create_engine(self.db_url, future=True)
        return self._engine

    def get_schema(self) -> Dict[str, List[str]]:
        """Return a mapping table -> [columns]

        Uses SQLAlchemy inspector which works for all common dialects.
        """
        engine = self.get_engine()
        insp = inspect(engine)
        tables = {}
        for t in insp.get_table_names():
            cols = [c["name"] for c in insp.get_columns(t)]
            tables[t] = cols
        return tables

    def execute_select(self, sql: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a SELECT statement and return rows as list of dicts.

        This wrapper assumes sql has been validated to be a SELECT.
        """
        engine = self.get_engine()
        logger.debug("Executing SQL: %s", sql)
        with engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            # Convert to list of dicts
            rows = [dict(r._mapping) for r in result]
        return rows
