"""
SQL Output Validator

This module provides STRICT OUTPUT VALIDATION for LLM-generated SQL.

PURPOSE:
- Validate SQL syntax, logic, and security AFTER LLM generation
- REJECT unsafe/invalid SQL - do not execute
- Only allow valid, safe SQL to proceed to execution

WORKFLOW:
LLM generates SQL → validate() → PASS (valid) or FAIL (invalid/unsafe)
- PASS: Continue to execution
- FAIL: Return error message, log incident, STOP workflow

No execution of invalid SQL. No bypassing validation.
"""

import re
import sqlparse
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationFailureReason(Enum):
    """Reasons why SQL validation might fail."""
    SYNTAX_ERROR = "syntax_error"
    SQL_INJECTION = "sql_injection"
    FORBIDDEN_OPERATION = "forbidden_operation"
    INVALID_STRUCTURE = "invalid_structure"
    EMPTY_QUERY = "empty_query"


@dataclass
class SQLValidationResult:
    """Result of SQL validation."""
    is_valid: bool
    sql_query: str
    normalized_sql: Optional[str]
    failure_reason: Optional[ValidationFailureReason]
    message: str
    errors: List[str]


class SQLOutputValidator:
    """
    Output validator that validates LLM-generated SQL queries.
    
    If SQL is invalid or unsafe, it is REJECTED and execution is prevented.
    Only valid, safe SQL proceeds to execution.
    """
    
    # Dangerous SQL patterns (injection attempts)
    INJECTION_PATTERNS = [
        r';\s*DROP\s+',
        r';\s*DELETE\s+',
        r';\s*TRUNCATE\s+',
        r';\s*ALTER\s+',
        r';\s*CREATE\s+',
        r';\s*EXEC\s*\(',
        r';\s*EXECUTE\s*\(',
        r'--\s*$',
        r'/\*.*\*/',
        r'xp_cmdshell',
        r'sp_executesql',
        r'UNION\s+ALL\s+SELECT',
        r'UNION\s+SELECT',
        r'OR\s+1\s*=\s*1',
        r'OR\s+\'1\'\s*=\s*\'1\'',
        r'\'\s+OR\s+\'\d\'\s*=\s*\'\d\'',
    ]
    
    # Forbidden keywords/operations
    FORBIDDEN_KEYWORDS = [
        'DROP', 'TRUNCATE', 'DELETE', 'INSERT', 'UPDATE',
        'ALTER', 'CREATE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE',
        'xp_', 'sp_', 'SHUTDOWN', 'BACKUP', 'RESTORE'
    ]
    
    # Allowed operations (read-only)
    ALLOWED_KEYWORDS = ['SELECT', 'WITH', 'FROM', 'WHERE', 'JOIN', 'GROUP', 'ORDER', 'HAVING', 'LIMIT']
    
    def __init__(self, strict_mode: bool = True, allow_subqueries: bool = True):
        """
        Initialize SQL output validator.
        
        Args:
            strict_mode: If True, reject on any validation failure
            allow_subqueries: Whether to allow subqueries in SQL
        """
        self.strict_mode = strict_mode
        self.allow_subqueries = allow_subqueries
        logger.info(f"SQLOutputValidator initialized (strict_mode={strict_mode})")
    
    def validate(self, sql_query: str) -> SQLValidationResult:
        """
        Validate LLM-generated SQL query.
        
        Args:
            sql_query: The SQL query to validate
            
        Returns:
            SQLValidationResult with is_valid=True (pass) or False (reject)
        """
        errors = []
        
        # Check for empty query
        if not sql_query or not sql_query.strip():
            return SQLValidationResult(
                is_valid=False,
                sql_query=sql_query,
                normalized_sql=None,
                failure_reason=ValidationFailureReason.EMPTY_QUERY,
                message="SQL validation FAILED: Empty query",
                errors=["Query is empty or whitespace only"]
            )
        
        # Quick check: detect comment-based injection patterns before normalization
        if re.search(r"--|/\*|\*/", sql_query):
            errors.append("Comment markers detected in SQL")
            logger.error("Comment-based injection detected in SQL")
            return SQLValidationResult(
                is_valid=False,
                sql_query=sql_query,
                normalized_sql=None,
                failure_reason=ValidationFailureReason.SQL_INJECTION,
                message="SQL validation FAILED: Comment-based injection detected",
                errors=errors,
            )

        # Normalize SQL (strip comments after we've already checked for them)
        try:
            normalized = sqlparse.format(
                sql_query,
                strip_comments=True,
                reindent=True,
                keyword_case='upper'
            ).strip()
        except Exception as e:
            logger.error(f"SQL parsing error: {e}")
            return SQLValidationResult(
                is_valid=False,
                sql_query=sql_query,
                normalized_sql=None,
                failure_reason=ValidationFailureReason.SYNTAX_ERROR,
                message=f"SQL validation FAILED: Parse error - {str(e)}",
                errors=[f"Parse error: {str(e)}"]
            )

        # 1. Check for SQL injection patterns (semicolons, suspicious functions)
        injection_detected = self._check_injection(normalized)
        if injection_detected:
            errors.append(f"SQL injection pattern detected: {injection_detected}")
            logger.error(f"SQL INJECTION DETECTED: {injection_detected}")
            error_detail = "semicolon and multiple statements" if ";" in sql_query else "SQL injection"
            return SQLValidationResult(
                is_valid=False,
                sql_query=sql_query,
                normalized_sql=None,
                failure_reason=ValidationFailureReason.SQL_INJECTION,
                message=f"SQL validation FAILED: Potential SQL injection detected ({error_detail})",
                errors=errors,
            )

        # 2. Check for forbidden operations (INSERT/UPDATE/DELETE/DROP/ALTER etc.)
        forbidden = self._check_forbidden_operations(normalized)
        if forbidden:
            errors.append(f"Forbidden operation detected: {forbidden}")
            logger.error(f"FORBIDDEN OPERATION: {forbidden}")
            return SQLValidationResult(
                is_valid=False,
                sql_query=sql_query,
                normalized_sql=None,
                failure_reason=ValidationFailureReason.FORBIDDEN_OPERATION,
                message=f"SQL validation FAILED: Forbidden operation '{forbidden}' - only SELECT queries allowed",
                errors=errors,
            )
        
        # 3. Validate SQL structure
        structure_errors = self._validate_structure(normalized)
        if structure_errors:
            errors.extend(structure_errors)
            logger.error(f"SQL structure validation failed: {structure_errors}")
            return SQLValidationResult(
                is_valid=False,
                sql_query=sql_query,
                normalized_sql=normalized,
                failure_reason=ValidationFailureReason.INVALID_STRUCTURE,
                message=f"SQL validation FAILED: Invalid structure",
                errors=errors
            )
        # 4. Detect inline string literals (unparameterized values) - discourage PII exposure
        #    Require models to use named parameters like :param_name for user-provided values
        #    However, allow small, harmless literals that are commonly used in SQL functions
        #    (e.g., ',', '.', numeric strings like '0', short format tokens). This avoids
        #    false positives for safe formatting operations such as REPLACE(..., ',', '.')
        inline_literals = re.findall(r"'([^']*)'", normalized)
        if inline_literals:
            def _is_allowed_literal(lit: str) -> bool:
                # empty string and strftime patterns are allowed
                if not lit or '%' in lit:
                    return True
                # numeric-like literals: 0, 1, 3.14, -1
                if re.match(r'^-?\d+(?:\.\d+)?$', lit):
                    return True
                # very short punctuation tokens used in REPLACE/formatting (e.g. ',', '.')
                if len(lit) <= 2 and re.match(r'^[\.,;:\-]$', lit):
                    return True
                return False

            non_allowed = [lit for lit in inline_literals if not _is_allowed_literal(lit)]
            if non_allowed:
                errors.append(f"Inline string literal(s) detected: {non_allowed[:5]}")
                logger.error(f"INLINE LITERAL(S) DETECTED: {non_allowed[:5]}")
                return SQLValidationResult(
                    is_valid=False,
                    sql_query=sql_query,
                    normalized_sql=normalized,
                    failure_reason=ValidationFailureReason.INVALID_STRUCTURE,
                    message=("SQL validation FAILED: Inline string literal(s) found. "
                             "All user-provided values must be parameterized using named parameters like :param_name."),
                    errors=errors,
                )

        # 5. Validate syntax using sqlparse
        syntax_errors = self._validate_syntax(normalized)
        if syntax_errors:
            errors.extend(syntax_errors)
            logger.error(f"SQL syntax validation failed: {syntax_errors}")
            return SQLValidationResult(
                is_valid=False,
                sql_query=sql_query,
                normalized_sql=normalized,
                failure_reason=ValidationFailureReason.SYNTAX_ERROR,
                message=f"SQL validation FAILED: Syntax errors detected",
                errors=errors,
            )

        # All checks passed - VALID SQL
        logger.info("SQL validation PASSED: Query is valid and safe")
        return SQLValidationResult(
            is_valid=True,
            sql_query=sql_query,
            normalized_sql=normalized,
            failure_reason=None,
            message="SQL is valid and safe",
            errors=[],
        )
    
    def _check_injection(self, sql: str) -> Optional[str]:
        """Check for SQL injection patterns."""
        upper_sql = sql.upper()
        
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, upper_sql, re.IGNORECASE):
                return pattern
        
        return None
    
    def _check_forbidden_operations(self, sql: str) -> Optional[str]:
        """Check for forbidden SQL operations."""
        upper_sql = sql.upper()
        
        # Check each statement in the query
        statements = sqlparse.split(sql)
        for statement in statements:
            statement_upper = statement.upper().strip()
            
            # Must start with SELECT or WITH (for CTEs)
            if not (statement_upper.startswith('SELECT') or statement_upper.startswith('WITH')):
                # Find the first keyword
                for keyword in self.FORBIDDEN_KEYWORDS:
                    if statement_upper.startswith(keyword):
                        return keyword
        
        # Check for forbidden keywords anywhere
        for keyword in self.FORBIDDEN_KEYWORDS:
            # Use word boundaries to avoid false positives
            pattern = r'\b' + keyword + r'\b'
            if re.search(pattern, upper_sql):
                return keyword
        
        return None
    
    def _validate_structure(self, sql: str) -> List[str]:
        """Validate SQL structure."""
        errors = []
        
        # Parse SQL
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                errors.append("Unable to parse SQL structure")
                return errors
            
            # Check for multiple statements (should be single query)
            if len(parsed) > 1:
                errors.append(f"Multiple SQL statements detected ({len(parsed)}). Only single SELECT queries allowed.")
            
            # Check statement type
            stmt = parsed[0]
            if stmt.get_type() not in ('SELECT', 'UNKNOWN'):
                errors.append(f"Invalid statement type: {stmt.get_type()}. Only SELECT queries allowed.")
            
        except Exception as e:
            errors.append(f"Structure validation error: {str(e)}")
        
        return errors
    
    def _validate_syntax(self, sql: str) -> List[str]:
        """Validate SQL syntax."""
        errors = []
        
        try:
            # Basic syntax checks
            if sql.count('(') != sql.count(')'):
                errors.append("Mismatched parentheses")
            
            if sql.count("'") % 2 != 0:
                errors.append("Unclosed string literal")
            
            # Check for required keywords in SELECT
            if 'SELECT' in sql.upper():
                if 'FROM' not in sql.upper():
                    # FROM is optional for simple selects like SELECT 1
                    pass
            
        except Exception as e:
            errors.append(f"Syntax validation error: {str(e)}")
        
        return errors
    
    def validate_strict(self, sql_query: str) -> Tuple[bool, str, Optional[str]]:
        """
        Strict validation that returns simple pass/fail.
        
        Args:
            sql_query: The SQL query to validate
            
        Returns:
            Tuple of (is_valid, message, normalized_sql)
            - (True, "valid", normalized_sql) if SQL is valid
            - (False, "rejection reason", None) if SQL is invalid
        """
        result = self.validate(sql_query)
        return result.is_valid, result.message, result.normalized_sql


# Convenience function for quick validation
def validate_sql_output(sql_query: str, strict: bool = True) -> Tuple[bool, str, Optional[str]]:
    """
    Quick validation function.
    
    Args:
        sql_query: SQL to validate
        strict: Whether to use strict validation
        
    Returns:
        (is_valid, message, normalized_sql) tuple
    """
    validator = SQLOutputValidator(strict_mode=strict)
    return validator.validate_strict(sql_query)
