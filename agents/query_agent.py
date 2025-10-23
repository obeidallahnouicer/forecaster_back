"""
agents/query_agent.py

Minimal QueryAgent implementing strict validation pipeline.

This file intentionally keeps the LLM usage minimal and uses lazy imports
so running unit tests or importing the module doesn't require LangChain/Groq
to be installed. The agent returns structured dicts indicating where
validation failed or succeeded.
"""

from typing import Dict, Any
import logging

from core import config
from core.schema_loader import get_schema_snapshot
from guardrails.pii_detector import PIIInputValidator
from guardrails.sql_validator import SQLOutputValidator
from prompts.insight_generation import generate_file_schema
import re
from core.db_connection import execute_select


def _extract_identifiers_from_sql(sql: str) -> set:
    """Naive SQL identifier extractor: returns a set of candidate column/table names.

    This is intentionally simple: it extracts word-like tokens and filters
    out common SQL keywords and functions. Good enough for validation step.
    """
    # Find tokens in SELECT, GROUP BY, ORDER BY clauses as likely column references
    cols = set()
    # capture between SELECT and FROM
    m = re.search(r"select(.*?)from", sql, flags=re.IGNORECASE | re.S)
    if m:
        select_part = m.group(1)
        cand = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ_][A-Za-z0-9_À-ÖØ-öø-ÿ]*", select_part)
        cols.update(cand)

    # capture GROUP BY and ORDER BY lists
    for clause in (r"group\s+by(.*?)(order|limit|having|$)", r"order\s+by(.*?)(limit|$)"):
        for mm in re.finditer(clause, sql, flags=re.IGNORECASE | re.S):
            part = mm.group(1)
            cand = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ_][A-Za-z0-9_À-ÖØ-öø-ÿ]*", part)
            cols.update(cand)

    # clean keywords and functions
    keywords = {
        'sum', 'count', 'avg', 'min', 'max', 'distinct', 'case', 'when', 'then', 'end'
    }
    return {t for t in cols if t.lower() not in keywords}

logger = logging.getLogger(__name__)


class QueryAgent:
    """Agent that generates SQL from natural language with strict validations."""

    def __init__(self):
        self.pii_validator = PIIInputValidator(strict_mode=True)
        self.sql_validator = SQLOutputValidator(strict_mode=True)
        schema_snapshot = get_schema_snapshot(limit_sample=1)
        # Convert schema snapshot to simple text format for LLM
        self.schema = self._format_schema(schema_snapshot)
        # Try to load prompt template if available
        try:
            with open("prompts/sql_generation.txt", "r", encoding="utf-8") as f:
                self.prompt_template = f.read()
        except Exception:
            self.prompt_template = None

    def generate_sql(self, question: str) -> Dict[str, Any]:
        """Run the validation pipeline and attempt SQL generation.

        Returns a dict with structured fields describing success/failure and
        validation stage where a failure occurred.
        """
        logger.info("Processing query")

        # Stage 1: Input validation (PII)
        pii_result = self.pii_validator.validate(question)
        if not pii_result.is_safe:
            logger.error("Input validation failed: PII detected")
            return {
                "success": False,
                "validation_stage": "input_validation",
                "error": pii_result.message,
                "detected_pii": pii_result.detected_types,
            }

        # Stage 2: LLM generation (lazy import)
        sql_text = None
        if not config.GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not set - LLM disabled")
            return {
                "success": False,
                "validation_stage": "llm_generation",
                "error": "LLM not configured (missing GROQ_API_KEY)",
            }

        try:
            # Lazy import to avoid hard dependency at module import time
            from langchain_core.prompts import PromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            from langchain_groq import ChatGroq

            # generate file-schema to provide machine-readable columns for validation
            file_schema = generate_file_schema()

            prompt = self.prompt_template or (
                """You are an expert SQL generator. Schema:\n{schema}\nFiles and columns:\n{file_schema}\nQuestion: {question}\nReturn only a single SELECT SQL query."""
            )

            # Build a very small chain using ChatGroq directly; we keep this
            # simple: call the LLM with a formatted prompt and take the output.
            llm = ChatGroq(model=config.GROQ_MODEL, api_key=config.GROQ_API_KEY, temperature=0.1)
            formatted = prompt.format(schema=self.schema, question=question, file_schema=file_schema)
            # Use invoke() instead of direct call - LangChain's new API
            response = llm.invoke(formatted)
            # Extract content from the response
            sql_text = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.exception("LLM generation error")
            return {
                "success": False,
                "validation_stage": "llm_generation",
                "error": f"LLM generation failed: {e}",
            }

        # Clean the LLM output
        sql_text = self._clean_sql(sql_text)
        # Stage 3: Output validation (SQL)
        sql_result = self.sql_validator.validate(sql_text)
        if not sql_result.is_valid:
            logger.error("Output validation failed for generated SQL")
            return {
                "success": False,
                "validation_stage": "output_validation",
                "error": sql_result.message,
                "sql_errors": sql_result.errors,
                "failure_reason": sql_result.failure_reason.value if sql_result.failure_reason else None,
                "generated_sql": sql_text,
            }

        # Attempt to execute the SQL so the caller can see actual results (safe in-memory DB)
        try:
            exec_res = execute_select(sql_result.normalized_sql, params=None, max_rows=10)
            return {
                "success": True,
                "validation_stage": "completed",
                "sql": sql_result.normalized_sql,
                "original_sql": sql_text,
                "message": "SQL generated, validated and executed",
                "params": {},
                "explanation": f"Generated SQL query for: {question}",
                "columns": exec_res.get('columns', []),
                "rows_preview": exec_res.get('rows', [])[:10],
                "rowcount": exec_res.get('rowcount', 0),
            }
        except Exception as e:
            logger.exception("Execution failed for generated SQL")
            return {
                "success": False,
                "validation_stage": "execution",
                "error": str(e),
                "generated_sql": sql_result.normalized_sql,
            }

    def _clean_sql(self, sql: str) -> str:
        if not sql:
            return sql
        # Remove markdown fences and common prefixes
        sql = sql.replace("```sql", "").replace("```", "")
        prefixes = ["SQL:", "Query:", "SELECT:"]
        s = sql.strip()
        for p in prefixes:
            if s.upper().startswith(p.upper()):
                s = s[len(p):].strip()
        # Normalize whitespace
        s = " ".join(s.split())
        return s

    def _format_schema(self, schema_snapshot: Dict[str, Any]) -> str:
        """Convert schema snapshot to simple text format for LLM."""
        lines = ["Available tables:"]
        for table_name, info in schema_snapshot.get('tables', {}).items():
            columns = info.get('columns', [])
            lines.append(f"\n{table_name}:")
            lines.append(f"  Columns: {', '.join(columns[:20])}")  # Limit to first 20 columns
            if info.get('samples'):
                sample = info['samples'][0]
                sample_str = ", ".join([f"{k}={repr(v)[:30]}" for k, v in list(sample.items())[:3]])
                lines.append(f"  Example: {sample_str}")
        return "\n".join(lines)


def generate_sql_from_question(question: str) -> Dict[str, Any]:
    """Convenience function to generate SQL from a natural language question."""
    agent = QueryAgent()
    return agent.generate_sql(question)


# Alias for backward compatibility
generate_sql = generate_sql_from_question
    
