"""
Guardrails Module

Strict input/output validation for SQL chatbot:

INPUT VALIDATORS (pre-LLM):
- PII detection: Blocks unsafe user input before sending to LLM

OUTPUT VALIDATORS (post-LLM):
- SQL syntax validation: Validates generated SQL structure
- SQL injection prevention: Blocks malicious SQL patterns
- Logic validation: Ensures safe query execution

Workflow:
1. User input → PII validator → REJECT if unsafe
2. Safe input → LLM → SQL generation
3. Generated SQL → SQL validator → REJECT if invalid/unsafe
4. Valid SQL → Execution → Results
"""

from .pii_detector import PIIInputValidator
from .sql_validator import SQLOutputValidator

__all__ = [
    'PIIInputValidator',      # Input validator
    'SQLOutputValidator',     # Output validator
]
