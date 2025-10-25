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
from prompts.sql_generation import SQL_SYSTEM_PROMPT, SQL_GENERATION_TEMPLATE
import re
from core.db_connection import execute_select
from typing import Any
import datetime
import sqlite3


def _extract_identifiers_from_sql(sql: str) -> set:
    """Naive SQL identifier extractor: returns a set of candidate column/table names.

    This is intentionally simple: it extracts word-like tokens and filters
    out common SQL keywords and functions. Good enough for validation step.
    """
    # Collect candidate identifiers from common SQL clauses: SELECT, GROUP BY, ORDER BY
    cols = set()

    # capture between SELECT and FROM (likely column list)
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

    # capture table names in FROM and JOIN clauses (including alias forms)
    # e.g., FROM t_ventes_cleann v, JOIN t_stock s ON ...
    for mm in re.finditer(r"(?:from|join)\s+([A-Za-z0-9_\.\"]+)", sql, flags=re.IGNORECASE):
        token = mm.group(1)
        # strip possible schema qualifiers or quoted identifiers
        token = token.strip(' ,')
        # if token contains a dot (schema.table), take last part
        if '.' in token:
            token = token.split('.')[-1]
        # remove surrounding quotes
        token = token.strip('"')
        # strip any trailing alias (handled separately by regex above, but be safe)
        token = re.sub(r"\s+as\s+.*$", "", token, flags=re.IGNORECASE)
        # split on whitespace to remove inline alias
        token = token.split()[0]
        if token:
            # allow tokens like t_ventes_cleann
            cols.add(token)

    # clean keywords and functions
    keywords = {
        'sum', 'count', 'avg', 'min', 'max', 'distinct', 'case', 'when', 'then', 'end',
        'select', 'from', 'where', 'group', 'by', 'order', 'limit', 'join', 'on', 'as'
    }
    # also ignore ordering keywords and common alias markers
    keywords.update({'desc', 'asc'})

    # Filter tokens and return
    return {t for t in cols if t and t.lower() not in keywords}

logger = logging.getLogger(__name__)


class QueryAgent:
    """Agent that generates SQL from natural language with strict validations."""

    def __init__(self):
        self.pii_validator = PIIInputValidator(strict_mode=True)
        self.sql_validator = SQLOutputValidator(strict_mode=True)
        
        # Use enhanced table schemas with business rules
        try:
            from core.table_schemas import format_schema_for_llm
            self.schema = format_schema_for_llm()
        except ImportError:
            # Fallback to old schema loading
            schema_snapshot = get_schema_snapshot(limit_sample=1)
            self.schema = self._format_schema(schema_snapshot)
        
        # Use the template from sql_generation.py
        self.prompt_template = SQL_GENERATION_TEMPLATE

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
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_groq import ChatGroq

            # generate file-schema to provide machine-readable columns for validation
            file_schema = generate_file_schema()

            # Provide a machine-readable JSON schema block expected by the prompt
            try:
                from prompts.sql_generation import build_json_schema_from_snapshot
                json_schema = build_json_schema_from_snapshot(limit_sample=1)
            except Exception:
                json_schema = None

            # Build chat prompt with system message + human template
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", SQL_SYSTEM_PROMPT),
                ("human", self.prompt_template)
            ])

            # Build a very small chain using ChatGroq directly
            llm = ChatGroq(model=config.GROQ_MODEL, api_key=config.GROQ_API_KEY, temperature=0.1)
            formatted = chat_prompt.format_messages(schema=self.schema, question=question, file_schema=file_schema, json_schema=json_schema)
            # Use invoke() with the formatted messages
            response = llm.invoke(formatted)
            # Extract content from the response
            sql_text = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.exception("LLM generation error")
            # Try a lightweight heuristic fallback for simple queries when LLM fails
            try:
                fallback = self._heuristic_sql_from_question(question)
                if fallback:
                    sql_text = fallback
                else:
                    return {
                        "success": False,
                        "validation_stage": "llm_generation",
                        "error": f"LLM generation failed: {e}",
                    }
            except Exception:
                return {
                    "success": False,
                    "validation_stage": "llm_generation",
                    "error": f"LLM generation failed: {e}",
                }

        # Clean the LLM output
        sql_text = self._clean_sql(sql_text)
        
        # Try to parse JSON response (LLM should return {"sql": "...", "params": {...}, "explanation": "..."})
        sql_params = {}
        try:
            import json
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', sql_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed, dict) and 'sql' in parsed:
                    sql_text = parsed.get('sql', '')
                    sql_params = parsed.get('params', {})
                    explanation = parsed.get('explanation', '')
        except Exception:
            pass  # If JSON parsing fails, use sql_text as-is

        # Post-process SQL to parameterize user-provided interval literals like INTERVAL '6 month'
        try:
            sql_text, sql_params = self._parameterize_intervals(sql_text, sql_params)
        except Exception:
            # don't block on parameterization errors; proceed to validation which will catch issues
            pass
        
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
        # Additional strict check: ensure the generated SQL references only tables/columns
        # that exist in the live schema snapshot. This prevents executing SQL that uses
        # invented tables (e.g., t_clients) or misspelled/accented column names.
        try:
            snap = get_schema_snapshot(limit_sample=0)
            known_tables = {t.lower(): {c.lower() for c in info.get('columns', [])} for t, info in snap.get('tables', {}).items()}
            identifiers = _extract_identifiers_from_sql(sql_result.normalized_sql or sql_text)
            unknown_identifiers = []
            for ident in identifiers:
                low = ident.lower()
                # identifier might be a table or a column; check both
                if low in known_tables:
                    continue
                if any(low in cols for cols in known_tables.values()):
                    continue
                # otherwise unknown
                unknown_identifiers.append(ident)

            if unknown_identifiers:
                msg = f"Generated SQL references unknown identifiers: {unknown_identifiers}.\nPlease regenerate using only the exact table and column names from the schema block."
                logger.error(msg)

                # Attempt one-shot automatic re-prompt: ask the LLM to rewrite the SQL
                # using only the exact physical table and column names provided in the
                # schema block. This gives the LLM a chance to correct hallucinated
                # identifiers without creating compatibility tables.
                if config.GROQ_API_KEY:
                    try:
                        # prepare machine-readable schema and file schema for the rewrite prompt
                        file_schema = generate_file_schema()
                        try:
                            from prompts.sql_generation import build_json_schema_from_snapshot
                            json_schema = build_json_schema_from_snapshot(limit_sample=1)
                        except Exception:
                            json_schema = None

                        # Lazy import LLM classes
                        from langchain_core.prompts import ChatPromptTemplate
                        from langchain_groq import ChatGroq

                        rewrite_system = (
                            "You were asked to produce a single SELECT statement in JSON form. "
                            "The previous SQL referenced identifiers that do not exist in the database. "
                            "Here is the exact schema (human readable):\n" + self.schema + "\n\n"
                        )

                        rewrite_human = (
                            "Please rewrite the SQL to produce the same intent but use ONLY the exact "
                            "table and column names present in the schema block below. Return a JSON object with keys: \"sql\", \"params\" (object) and \"explanation\" (string). "
                            "Do not invent tables or columns. If you cannot express the intent using the schema, return an empty sql string.\n\n"
                            f"Schema (human):\n{self.schema}\n\n"
                        )

                        # include machine-readable json schema when available
                        if json_schema:
                            rewrite_human += f"Schema (json):\n{json_schema}\n\n"

                        rewrite_human += f"Original SQL: {sql_text}\n\nOriginal question: {question}"

                        # Avoid template formatting of raw JSON by passing plain messages
                        llm = ChatGroq(model=config.GROQ_MODEL, api_key=config.GROQ_API_KEY, temperature=0.0)
                        messages = [
                            {"role": "system", "content": rewrite_system},
                            {"role": "user", "content": rewrite_human},
                        ]
                        resp = llm.invoke(messages)
                        new_text = resp.content if hasattr(resp, 'content') else str(resp)
                        new_text = self._clean_sql(new_text)

                        # Attempt to parse JSON from the rewrite response
                        try:
                            jmatch = re.search(r'\{.*\}', new_text, re.DOTALL)
                            if jmatch:
                                import json as _json
                                parsed = _json.loads(jmatch.group(0))
                                if isinstance(parsed, dict) and 'sql' in parsed:
                                    sql_text = parsed.get('sql', '')
                                    sql_params = parsed.get('params', {}) or {}
                                    explanation = parsed.get('explanation', '')
                        except Exception:
                            logger.exception("Failed to parse rewritten SQL JSON from LLM")

                        # Re-run syntactic validation on rewritten SQL
                        sql_result = self.sql_validator.validate(sql_text)
                        if not sql_result.is_valid:
                            logger.error("Rewritten SQL failed validation")
                            # fall through to return the original validation error below
                        else:
                            # Re-check identifiers against snapshot; if clean, continue execution
                            snap2 = get_schema_snapshot(limit_sample=0)
                            known_tables2 = {t.lower(): {c.lower() for c in info.get('columns', [])} for t, info in snap2.get('tables', {}).items()}
                            identifiers2 = _extract_identifiers_from_sql(sql_result.normalized_sql or sql_text)
                            unknown2 = [i for i in identifiers2 if not (i.lower() in known_tables2 or any(i.lower() in cols for cols in known_tables2.values()))]
                            if not unknown2:
                                # Accept rewritten SQL
                                logger.info("LLM rewrite produced SQL using known identifiers; proceeding to execution")
                                # update sql_result and continue on to execution path
                                # Note: sql_result already set for rewritten SQL
                                pass
                            else:
                                logger.error("Rewritten SQL still references unknown identifiers: %s", unknown2)
                    except Exception:
                        logger.exception("Automatic LLM rewrite attempt failed")

                # If we reach here and haven't accepted a rewritten SQL, return the validation error
                # Attempt a local heuristic fix before failing: build a safe SQL using known tables/columns
                try:
                    local_fix = self._local_fix_for_unknown_identifiers(sql_text, question)
                    if local_fix:
                        # validate and, if valid, execute the fix
                        val = self.sql_validator.validate(local_fix)
                        if val.is_valid:
                            mapped_sql = self._map_logical_table_names(val.normalized_sql or local_fix)
                            try:
                                exec_res = execute_select(mapped_sql, params={}, max_rows=10)
                                return {
                                    "success": True,
                                    "validation_stage": "completed",
                                    "sql": mapped_sql,
                                    "original_sql": sql_text,
                                    "message": "SQL generated, validated and executed (local heuristic fix applied)",
                                    "params": {},
                                    "explanation": f"Local heuristic correction applied for: {question}",
                                    "columns": exec_res.get('columns', []),
                                    "rows_preview": exec_res.get('rows', [])[:10],
                                    "rowcount": exec_res.get('rowcount', 0),
                                }
                            except Exception:
                                # if execution fails, fall through to return error
                                logger.exception("Execution of local heuristic fix failed")
                except Exception:
                    logger.exception("Local heuristic fix attempt failed")

                return {
                    "success": False,
                    "validation_stage": "output_validation",
                    "error": msg,
                    "generated_sql": sql_text,
                    "unknown_identifiers": unknown_identifiers,
                }
        except Exception:
            # If snapshot check fails for some reason, don't block execution - rely on DB errors.
            logger.exception("Identifier existence check failed; proceeding to execution and relying on DB errors")

        # Attempt to execute the SQL so the caller can see actual results (safe in-memory DB)
        try:
            # Normalize parameter keys: allow keys with or without leading ':' from the LLM
            normalized_params = { (k[1:] if k.startswith(":") else k): v for k, v in (sql_params or {}).items() }

            # Extract named params used in the SQL (e.g., :client_name)
            required_params = set(re.findall(r":([A-Za-z0-9_]+)", sql_result.normalized_sql or ""))

            # If any required parameter is missing, return a clear execution error
            missing = [p for p in required_params if p not in normalized_params]
            if missing:
                msg = f"Missing parameter values for: {missing}. Provide values for these named parameters."
                logger.error(msg)
                return {
                    "success": False,
                    "validation_stage": "execution",
                    "error": msg,
                    "generated_sql": sql_result.normalized_sql,
                    "params": normalized_params,
                }

            # Map logical table names (e.g., 'ventes_cleann') to actual DB table names
            # (e.g., 't_ventes_cleann') before executing the query.
            mapped_sql = self._map_logical_table_names(sql_result.normalized_sql or sql_text)
            # Log the SQL that will be executed (debug level) to ease debugging syntax errors
            logger.debug("Executing mapped SQL: %s", mapped_sql)

            # Execute and catch DB-specific errors so we return a structured response
            try:
                exec_res = execute_select(mapped_sql, params=normalized_params, max_rows=10)
            except sqlite3.OperationalError as sqe:
                # Provide the mapped SQL and params in the structured error so callers
                # (and the API router) can return a meaningful message instead of 500.
                logger.exception("SQLite OperationalError while executing SQL")

                # Attempt an automatic fallback for missing tables: if the error
                # indicates a missing table (e.g., 'no such table: t_clients'),
                # try to find a candidate table in the schema snapshot that contains
                # client-like columns (Code_Client, Intitulé/Intitule_Client, Representant)
                msg = str(sqe)
                m = re.search(r"no such table:\s*([A-Za-z0-9_]+)", msg, flags=re.IGNORECASE)
                if m:
                    missing = m.group(1)
                    logger.info("Detected missing table '%s' - attempting fallback lookup in schema snapshot", missing)
                    try:
                        snap = get_schema_snapshot(limit_sample=0)
                        candidate = None
                        # prefer ventes table variant if available
                        for tname, info in snap.get('tables', {}).items():
                            cols = [c.lower() for c in info.get('columns', [])]
                            if 'code_client' in cols or 'code_client' in tname.lower():
                                candidate = tname
                                break

                        if not candidate:
                            # fallback: pick any table that has 'code_client' like column
                            for tname, info in snap.get('tables', {}).items():
                                cols = [c.lower() for c in info.get('columns', [])]
                                if any(k in cols for k in ('code_client', 'intitule_client', 'representant')):
                                    candidate = tname
                                    break

                        if candidate:
                            logger.info("Retrying query replacing missing table '%s' with candidate '%s'", missing, candidate)
                            replaced_sql = re.sub(r"\b" + re.escape(missing) + r"\b", candidate, mapped_sql, flags=re.IGNORECASE)
                            logger.debug("Rewritten SQL for retry: %s", replaced_sql)
                            try:
                                exec_res = execute_select(replaced_sql, params=normalized_params, max_rows=10)
                                # If successful, return the results (note we expose the rewritten SQL)
                                return {
                                    "success": True,
                                    "validation_stage": "completed",
                                    "sql": replaced_sql,
                                    "original_sql": sql_text,
                                    "message": f"SQL generated, validated and executed (rewritten replacing {missing} with {candidate})",
                                    "params": normalized_params,
                                    "explanation": f"Generated SQL query for: {question}",
                                    "columns": exec_res.get('columns', []),
                                    "rows_preview": exec_res.get('rows', [])[:10],
                                    "rowcount": exec_res.get('rowcount', 0),
                                }
                            except sqlite3.OperationalError as retry_sqe:
                                # If retry failed due to missing column, attempt to auto-derive common
                                # computed columns (e.g., Mois_Depuis_Derniere_Vente) from available
                                # sales Date column in the candidate table and retry.
                                rmsg = str(retry_sqe)
                                cm = re.search(r"no such column:\s*([A-Za-z0-9_]+)", rmsg, flags=re.IGNORECASE)
                                if cm:
                                    missing_col = cm.group(1)
                                    logger.info("Retry failed due to missing column '%s' - attempting to derive it", missing_col)
                                    # Known derived column: Mois_Depuis_Derniere_Vente -> compute months since Date
                                    if missing_col.lower() == 'mois_depuis_derniere_vente':
                                        # SQLite months-difference expression (integer months)
                                        months_expr = (
                                            "( (strftime('%Y','now') - strftime('%Y', Date)) * 12 + "
                                            "(strftime('%m','now') - strftime('%m', Date)) )"
                                        )
                                        derived_sql = re.sub(r"\b" + re.escape(missing_col) + r"\b", months_expr, replaced_sql, flags=re.IGNORECASE)
                                        logger.debug("Retrying with derived column expression: %s", derived_sql)
                                        try:
                                            exec_res = execute_select(derived_sql, params=normalized_params, max_rows=10)
                                            return {
                                                "success": True,
                                                "validation_stage": "completed",
                                                "sql": derived_sql,
                                                "original_sql": sql_text,
                                                "message": f"SQL generated, validated and executed (rewritten replacing {missing} with {candidate} and deriving {missing_col})",
                                                "params": normalized_params,
                                                "explanation": f"Generated SQL query for: {question}",
                                                "columns": exec_res.get('columns', []),
                                                "rows_preview": exec_res.get('rows', [])[:10],
                                                "rowcount": exec_res.get('rowcount', 0),
                                            }
                                        except Exception:
                                            logger.exception("Retry with derived column failed")
                                # if we can't handle it, log and continue to outer error return
                                logger.exception("Retry with candidate table failed (operational error)")
                            except Exception:
                                logger.exception("Retry with candidate table failed")

                    except Exception:
                        logger.exception("Fallback lookup failed")

                return {
                    "success": False,
                    "validation_stage": "execution",
                    "error": f"SQLite execution error: {sqe}",
                    "generated_sql": mapped_sql,
                    "params": normalized_params,
                }

            # Normal successful execution
            return {
                "success": True,
                "validation_stage": "completed",
                "sql": mapped_sql,
                "original_sql": sql_text,
                "message": "SQL generated, validated and executed",
                "params": normalized_params,
                "explanation": f"Generated SQL query for: {question}",
                "columns": exec_res.get('columns', []),
                "rows_preview": exec_res.get('rows', [])[:10],
                "rowcount": exec_res.get('rowcount', 0),
            }
        except Exception as e:
            logger.exception("Execution failed for generated SQL")
            err_msg = str(e)

            # If the error looks like a misuse of window functions (SQLite limitation),
            # attempt a safe retry by asking the LLM to rewrite the query without
            # window functions. This uses our llm/sql_agent adapter which already
            # formats prompts according to the project's rules.
            if 'window function' in err_msg.lower() or 'lag(' in err_msg.lower() or 'lead(' in err_msg.lower():
                try:
                    logger.info("Detected window function error; requesting LLM to rewrite without window functions")
                    from llm import sql_agent

                    rewrite_question = (
                        f"The previous SQL failed to execute in SQLite with error: {err_msg}. "
                        "Please rewrite the SQL to avoid using window functions (LAG/LEAD/ROW_NUMBER/etc.) "
                        "and instead use joins or subqueries. Keep to the allowed columns and return only a SELECT statement. "
                        f"Original user question: {question}"
                    )

                    # Use a minimal sample_rows (empty) — the prompt builder will include schema
                    rewrite_resp = sql_agent.generate_sql_from_question(rewrite_question, [])
                    if rewrite_resp.get('sql'):
                        # Return the rewritten SQL to the router for execution instead
                        # of executing it here. This keeps execution centralized in
                        # the API router/ExecutorAgent and avoids double execution
                        # paths that can lead to unexpected HTTP 500 responses.
                        rewritten = rewrite_resp['sql']
                        rewritten_mapped = self._map_logical_table_names(rewritten)
                        # Validate rewritten SQL before returning
                        val = self.sql_validator.validate(rewritten)
                        if not val.is_valid:
                            return {
                                "success": False,
                                "validation_stage": "execution",
                                "error": f"Rewritten SQL failed validation: {val.message}",
                                "generated_sql": rewritten,
                                "sql_errors": val.errors,
                            }

                        return {
                            "success": True,
                            # Mark as completed so the router will proceed to execution
                            # with the rewritten, validated SQL.
                            "validation_stage": "completed",
                            "sql": rewritten_mapped,
                            "params": rewrite_resp.get('params', {}),
                            "explanation": rewrite_resp.get('explanation', ''),
                            "original_sql": sql_text,
                            "rewritten_from_window_function": True,
                        }
                except Exception:
                    logger.exception("Failed to request LLM rewrite for window function error")

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

    def _heuristic_sql_from_question(self, question: str) -> str:
        """Very small rule-based fallback for common simple queries when LLM is unavailable.

        This is intentionally conservative and only covers basic patterns:
        - COUNT queries (how many / count)
        - SHOW/LIST queries (show/list/top N)
        Returns raw SQL string or empty string if no heuristic match.
        """
        if not question:
            return ""
        q = question.lower()

        # get available tables snapshot
        try:
            snap = get_schema_snapshot(limit_sample=0)
            tables = list(snap.get('tables', {}).keys())
        except Exception:
            tables = []

        # find candidate table by matching logical name in question
        candidate = None
        for t in tables:
            logical = t.lower()
            if logical.startswith('t_'):
                logical = logical[2:]
            if logical in q:
                candidate = t
                break
            # also match bare words like 'stock' with 't_stock'
            if logical.replace('_', ' ') in q:
                candidate = t
                break

        # fallback to common table names
        if not candidate:
            for common in ('t_stock', 't_ventes_cleann', 't_base'):
                if common in tables:
                    if any(k in q for k in ('stock', 'base', 'products')):
                        candidate = common
                        break

        if not candidate:
            return ""

        # Decide on SQL type
        if 'how many' in q or 'count' in q or 'how many' in q:
            return f"SELECT COUNT(*) as count FROM {candidate} LIMIT 1"

        # show / list top N
        m = re.search(r"(top|show|list)\s+(\d+)", q)
        if m:
            n = int(m.group(2))
            return f"SELECT * FROM {candidate} LIMIT {n}"

        if 'show' in q or 'list' in q or 'give me' in q:
            return f"SELECT * FROM {candidate} LIMIT 5"

        # no heuristic match
        return ""

    def _local_fix_for_unknown_identifiers(self, sql_text: str, question: str) -> str:
        """Attempt to build a safe SQL fallback locally when generated SQL references unknown identifiers.

        Conservative approach: pick a candidate table from the schema and use a stable key column
        (e.g., Ref_Article / Référence_Article) to compute counts or simple selects.
        """
        try:
            snap = get_schema_snapshot(limit_sample=0)
            tables = snap.get('tables', {})
        except Exception:
            return ""

        q = (question or "").lower()
        candidate = None
        for t in tables:
            logical = t.lower()
            if logical.startswith('t_'):
                logical = logical[2:]
            if logical in q:
                candidate = t
                break

        if not candidate:
            for pref in ('t_stock', 't_base', 't_ventes_cleann'):
                if pref in tables:
                    candidate = pref
                    break

        if not candidate:
            return ""

        cols = tables.get(candidate, {}).get('columns', [])
        if not cols:
            return ""

        # Prefer a reference-like column
        ref_col = None
        for c in cols:
            low = c.lower()
            if 'ref' in low or 'référence' in low or 'reference' in low or 'qte' in low or 'qt' in low:
                ref_col = c
                break
        if not ref_col:
            ref_col = cols[0]

        # Build a safe COUNT DISTINCT query
        return f'SELECT COUNT(DISTINCT "{ref_col}") AS count FROM {candidate} LIMIT 1'

    def _parameterize_intervals(self, sql: str, params: dict) -> tuple:
        """Detect SQL INTERVAL literals like INTERVAL '6 month' and replace them
        with a named parameter (e.g., :since_date). Returns (sql, params).

        This computes the since_date in Python (approximate month->30 days) and
        injects it into params so the SQL no longer contains inline string literals.
        """
        if not sql:
            return sql, params

        params = params or {}

        # match patterns like (CURRENT_DATE - INTERVAL '6 month') or INTERVAL '6 month'
        pattern = re.compile(r"\(\s*CURRENT_DATE\s*-\s*INTERVAL\s*'([^']+)'\s*\)", flags=re.IGNORECASE)

        def repl(match):
            token = match.group(1)  # e.g., "6 month"
            m = re.match(r"(\d+)\s*(month|months|day|days|year|years)", token.strip(), flags=re.IGNORECASE)
            if not m:
                # if we can't parse, keep original
                return match.group(0)
            qty = int(m.group(1))
            unit = m.group(2).lower()
            days = 0
            if unit.startswith('month'):
                days = qty * 30
            elif unit.startswith('year'):
                days = qty * 365
            else:
                # days
                days = qty

            since_date = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
            # choose a param name that is unlikely to collide
            param_name = 'since_date'
            # ensure unique param name if already present
            i = 1
            base = param_name
            while param_name in params:
                i += 1
                param_name = f"{base}_{i}"

            params[param_name] = since_date
            # return the named parameter for SQL (without quotes)
            return f":{param_name}"

        new_sql = pattern.sub(repl, sql)
        return new_sql, params

    def _map_logical_table_names(self, sql: str) -> str:
        """Map logical table names used by the LLM prompt to actual DB table names.

        The in-memory DB uses table names like `t_ventes_cleann` and `t_stock`. The
        LLM prompt and examples may refer to `ventes_cleann` or `STOCK`. This
        function replaces logical names with the concrete table names present in
        `self.schema` (produced by _format_schema) using word-boundary, case-
        insensitive replacement.
        """
        if not sql or not self.schema:
            return sql

        # Build mapping: logical_name -> actual_table_name
        mapping = {}
        for actual in self.schema.splitlines():
            # lines include e.g. '\nTableName:' and '  Columns: ...'
            # We instead read from schema snapshot stored earlier via get_schema_snapshot
            pass

        # Better: inspect the real schema snapshot again to build accurate mapping
        try:
            from core.schema_loader import get_schema_snapshot
            snap = get_schema_snapshot(limit_sample=0)
            for actual_name in snap.get('tables', {}).keys():
                # logical variants: strip leading 't_' or 'T_' if present
                if actual_name.lower().startswith('t_'):
                    logical = actual_name[2:]
                else:
                    logical = actual_name
                mapping[logical.lower()] = actual_name
                mapping[actual_name.lower()] = actual_name
        except Exception:
            # If snapshot lookup fails, return original SQL
            return sql

        # Replace occurrences of logical table names with actual table names
        def repl(m):
            token = m.group(0)
            key = token.lower()
            return mapping.get(key, token)

        # Build regex to match any of the logical keys as whole words
        keys = sorted(mapping.keys(), key=lambda x: -len(x))
        pattern = r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b"
        try:
            new_sql = re.sub(pattern, repl, sql, flags=re.IGNORECASE)
            return new_sql
        except Exception:
            return sql

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
    
