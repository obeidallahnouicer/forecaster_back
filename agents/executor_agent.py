"""
agents/executor_agent.py

ExecutorAgent executes validated SQL queries. It enforces that the
SQL passed in has already been validated by the output validator. If
validation_passed is False the execution will be refused.
"""

import logging
import time
from typing import Dict, Any
import pandas as pd

from core.db_connection import execute_select

logger = logging.getLogger(__name__)


class ExecutorAgent:
    def __init__(self):
        logger.info("ExecutorAgent initialized")

    def execute(self, sql_query: str, validation_passed: bool = False) -> Dict[str, Any]:
        if not validation_passed:
            logger.error("Execution blocked: SQL has not passed validation")
            return {"success": False, "error": "SQL execution rejected: validation_passed flag is False"}

        start = time.time()
        try:
            result = execute_select(sql_query, params=None, max_rows=1000)
            elapsed = (time.time() - start) * 1000

            rows = result.get('rows', [])
            columns = result.get('columns', [])
            row_count = result.get('rowcount', 0)

            logger.info(f"Query executed successfully: rows={row_count} time={elapsed:.1f}ms")
            return {"success": True, "data": rows, "row_count": row_count, "columns": columns, "execution_time_ms": elapsed}

        except Exception as e:
            logger.exception("SQL execution failed")
            return {"success": False, "error": str(e)}

    def execute_safe(self, sql_query: str) -> Dict[str, Any]:
        logger.info("SAFE EXECUTION REQUEST")
        return self.execute(sql_query, validation_passed=True)


def execute_query(sql: str, params: Dict[str, Any] = None, max_rows: int = 500) -> Dict[str, Any]:
    """
    Execute SQL query with optional parameters.
    
    Args:
        sql: Validated SQL query
        params: Query parameters (unused for now, kept for compatibility)
        max_rows: Maximum rows to return
    
    Returns:
        {
            'columns': List[str],
            'rows': List[Dict],
            'rowcount': int,
            'execution_time_ms': float
        }
    """
    start = time.time()
    
    # Execute query using core db_connection
    result = execute_select(sql, params=params, max_rows=max_rows)
    
    elapsed = (time.time() - start) * 1000
    result['execution_time_ms'] = elapsed
    
    logger.info(f"Executed SQL in {elapsed:.1f}ms, rows={result.get('rowcount', 0)}")
    
    return result
