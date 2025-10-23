"""
SQL Validator using Guardrails AI

Enhanced SQL validation with comprehensive safety checks:
- Syntax validation
- SQL injection prevention
- SELECT-only enforcement
- LIMIT enforcement
"""
import logging
from typing import Tuple

from guardrails.sql_validator import validate_sql_output

logger = logging.getLogger("agents.sql_validator")


def validate_sql(sql: str, schema_tables: set = None, enforce_limit: int = 1000) -> Tuple[bool, str, str]:
    """
    Validate SQL with Guardrails AI safety checks.
    
    Performs comprehensive validation:
    1. Syntax correctness
    2. SELECT-only operations
    3. SQL injection prevention
    4. LIMIT enforcement
    5. Dangerous keyword detection
    
    Args:
        sql: SQL query to validate
        schema_tables: (unused, kept for compatibility)
        enforce_limit: Maximum rows to return
    
    Returns:
        (valid, reason, adjusted_sql)
        - valid: True if query is safe
        - reason: Explanation of validation result
        - adjusted_sql: Modified SQL (e.g., with LIMIT added)
    """
    # Comprehensive validation with Guardrails
    is_valid, reason, adjusted_sql = validate_sql_output(sql, strict=True)
    
    if not is_valid:
        logger.error(f"SQL validation failed: {reason}")
        return False, reason, ''
    
    # Use the adjusted SQL from validator or original if none
    sql_to_adjust = adjusted_sql or sql
    
    # Add LIMIT if not present
    if 'LIMIT' not in sql_to_adjust.upper():
        # Add LIMIT at the end of the query
        sql_to_adjust = sql_to_adjust.strip()
        if sql_to_adjust.endswith(';'):
            sql_to_adjust = sql_to_adjust[:-1].strip()
        sql_to_adjust = f"{sql_to_adjust} LIMIT {enforce_limit}"
        logger.info(f"Added LIMIT {enforce_limit} to SQL")
    
    logger.info(f"SQL validation passed: {reason}")
    return True, reason, sql_to_adjust



