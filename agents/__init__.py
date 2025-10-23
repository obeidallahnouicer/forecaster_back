"""
Text-to-SQL Agents Package

Simple functional agents for the SQL-based chatbot workflow:
- QueryAgent: Natural language -> SQL generation
- SQLValidator: SQL safety validation
- ExecutorAgent: Safe SQL execution
- InsightAgent: Data-driven analysis and recommendations
"""

from .query_agent import generate_sql
from .sql_validator import validate_sql
from .executor_agent import execute_query
from .insight_agent import summarize_results

__all__ = [
    'generate_sql',
    'validate_sql',
    'execute_query',
    'summarize_results'
]
