"""
SQL Validation Prompts

Prompts used for SQL validation and safety checks.
"""

from langchain_core.prompts import PromptTemplate

VALIDATION_PROMPT_TEMPLATE = """You are a SQL security validator. 

Analyze this SQL query for safety and correctness:

SQL: {sql}

Check for:
1. Only SELECT operations (no INSERT/UPDATE/DELETE/DROP)
2. No SQL injection patterns (', --, /*, xp_, etc.)
3. No system table access
4. Proper SQLite syntax
5. No dangerous functions (LOAD_EXTENSION, etc.)

Return JSON:
{{
  "valid": true/false,
  "issues": ["list of issues if any"],
  "risk_level": "safe/warning/dangerous",
  "recommendation": "suggestion if needed"
}}"""

VALIDATION_PROMPT = PromptTemplate(
    input_variables=["sql"],
    template=VALIDATION_PROMPT_TEMPLATE
)
