"""
agents/query_agent.py

QueryAgent implementing strict validation pipeline for natural language to SQL conversion.

This file intentionally keeps the LLM usage minimal and uses lazy imports
so running unit tests or importing the module doesn't require LangChain/Groq
to be installed. The agent returns structured dicts indicating where
validation failed or succeeded.
"""

from typing import Dict, Any
import logging
import re
import datetime
import difflib
import unicodedata

from core import config
from core.schema_loader import get_schema_snapshot
from core.db_connection import execute_select
from guardrails.pii_detector import PIIInputValidator
from guardrails.sql_validator import SQLOutputValidator
from prompts.sql_generation import SQL_SYSTEM_PROMPT, SQL_GENERATION_TEMPLATE


def _extract_identifiers_from_sql(sql: str) -> set:
    """Naive SQL identifier extractor: returns a set of candidate column/table names.

    This is intentionally simple: it extracts word-like tokens and filters
    out common SQL keywords and functions. Good enough for validation step.
    """
    # Collect candidate identifiers from common SQL clauses: SELECT, GROUP BY, ORDER BY
    cols = set()

    # Collect alias names declared in the query so we can ignore them as identifiers.
    alias_names = set()
    for am in re.finditer(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)", sql, flags=re.IGNORECASE):
        alias_names.add(am.group(1))
    for am in re.finditer(r"\)[\s\n]*([A-Za-z_][A-Za-z0-9_]*)", sql):
        alias_names.add(am.group(1))

    # capture between SELECT and FROM (likely column list)
    m = re.search(r"select(.*?)from", sql, flags=re.IGNORECASE | re.S)
    if m:
        select_part = m.group(1)
        cand = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ_][A-Za-z0-9_À-ÖØ-öø-ÿ]*", select_part)
        cols.update(cand)
        # Remove aliases found in the whole SQL from the collected select tokens
        cols = cols.difference(alias_names)

    # capture GROUP BY and ORDER BY lists
    for clause in (r"group\s+by(.*?)(order|limit|having|$)", r"order\s+by(.*?)(limit|$)"):
        for mm in re.finditer(clause, sql, flags=re.IGNORECASE | re.S):
            part = mm.group(1)
            cand = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ_][A-Za-z0-9_À-ÖØ-öø-ÿ]*", part)
            cols.update(cand)
        # Remove any alias names from group/order tokens as they are not real table/column identifiers
        if alias_names:
            cols = cols.difference(alias_names)

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

        # Stage 2: LLM generation (STRICT: delegate to llm.sql_agent and accept ONLY its SQL)
        sql_text = None
        sql_params = {}
        explanation = ""

        try:
            from llm import sql_agent as llm_sql_agent
            resp = llm_sql_agent.generate_sql_from_question(question, sample_rows=[])
            explanation = resp.get("reasoning") or ""
            sql_text = resp.get("sql_query") or ""
            sql_params = resp.get("params") or {}

            # Enforce strict behavior: if adapter didn't return a SELECT, return error immediately
            if not sql_text:
                return {
                    "success": False,
                    "validation_stage": "llm_generation",
                    "error": "LLM did not return a valid SELECT query",
                    "llm_reasoning": explanation,
                    "llm_raw": resp.get("raw"),
                    "llm_meta": resp.get("meta"),
                }

        except Exception as e:
            logger.exception("LLM generation (adapter) error")
            return {
                "success": False,
                "validation_stage": "llm_generation",
                "error": f"LLM generation failed: {e}",
            }

        # Clean the SQL text produced by the adapter or fallback
        sql_text = self._clean_sql(sql_text)

        # If params are not present yet, try to extract JSON from the LLM text (legacy fallback)
        sql_params = sql_params or {}
        try:
            import json
            # Only attempt legacy JSON extraction if we don't already have params
            if not sql_params:
                json_match = re.search(r'\{.*\}', sql_text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    if isinstance(parsed, dict) and 'sql' in parsed:
                        sql_text = parsed.get('sql', '')
                        sql_params = parsed.get('params', {}) or {}
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
                msg = f"Generated SQL references unknown identifiers: {unknown_identifiers}."
                logger.error(msg)

                # Try a quick automatic fix using normalization + fuzzy matching before
                # invoking the LLM-driven retry flow. This often fixes issues like
                # accent differences, casing, or small typos (e.g., Intitulé_Client -> Intitule_Client).
                try:
                    def _normalize(s: str) -> str:
                        s = s or ''
                        # Unicode normalize and remove diacritics
                        s = unicodedata.normalize('NFKD', s)
                        s = ''.join(ch for ch in s if not unicodedata.combining(ch))
                        s = s.lower()
                        # replace non-alnum with underscore
                        s = re.sub(r'[^a-z0-9]+', '_', s)
                        s = s.strip('_')
                        return s

                    # Build candidate list: table names and columns flattened
                    flat_candidates = {}
                    for t, cols in known_tables.items():
                        flat_candidates[t] = t
                        for c in cols:
                            flat_candidates[c] = c

                    mappings = {}
                    for ident in list(unknown_identifiers):
                        norm = _normalize(ident)
                        # exact normalized match
                        found = None
                        for cand in flat_candidates.keys():
                            if _normalize(cand) == norm:
                                found = flat_candidates[cand]
                                break
                        # fuzzy match fallback
                        if not found:
                            choices = list(flat_candidates.keys())
                            close = difflib.get_close_matches(ident, choices, n=1, cutoff=0.7)
                            if close:
                                found = flat_candidates[close[0]]

                        if found and found != ident:
                            mappings[ident] = found

                    # If we found mappings, apply textual replacements (word-boundary)
                    if mappings:
                        new_sql = sql_result.normalized_sql or sql_text
                        for orig, replacement in mappings.items():
                            # replace occurrences with exact replacement (preserve case of replacement)
                            new_sql = re.sub(rf"\b{re.escape(orig)}\b", replacement, new_sql)

                        logger.info("Auto-corrected identifiers using fuzzy mapping: %s", mappings)
                        # Update sql_text and continue without triggering LLM retries
                        sql_text = new_sql
                        sql_params = sql_params or {}
                    else:
                        # If no mapping found, fall back to LLM retry mechanism below
                        pass

                except Exception:
                    # If auto-fix failed for any reason, continue to LLM-driven retries
                    logger.exception("Auto identifier-fix failed; falling back to LLM retries")

                # Attempt to have the LLM regenerate corrected SQL using only the schema
                # Try a limited number of retries where we ask the model to fix the query.
                try:
                    # Ensure we have access to the adapter
                    try:
                        from llm import sql_agent as llm_sql_agent
                    except Exception:
                        llm_sql_agent = None

                    # Read max retries from config (fallback to 2)
                    max_retries = getattr(config, 'LLM_RETRY_ATTEMPTS', 2)
                    retry_attempts = []
                    fixed_sql = None
                    fixed_params = {}
                    fixed_reasoning = None

                    # Keep the original failing SQL and reasoning to show the model what to fix
                    original_sql = sql_result.normalized_sql or sql_text
                    original_reasoning = explanation or ""

                    for attempt in range(max_retries):
                        if llm_sql_agent is None:
                            break

                        # Try to include the project's TABLE Chatbot.md as contextual schema/business rules
                        table_md = None
                        try:
                            # Project root is two levels up from this file
                            repo_root = Path(__file__).resolve().parents[1]
                            md_path = repo_root / "TABLE Chatbot.md"
                            if md_path.exists():
                                table_md = md_path.read_text(encoding="utf-8")
                        except Exception:
                            table_md = None

                        # Build retry note: include failing identifiers, the failing SQL and prior reasoning,
                        # plus the project's TABLE Chatbot.md (larger excerpt) and the original raw LLM output
                        # so the model can see exactly what was returned and how to fix it.
                        md_excerpt = None
                        if table_md:
                            # include more of the MD file (up to 20k chars) to give richer context
                            md_excerpt = table_md if len(table_md) <= 20000 else table_md[:20000] + "\n...\n"
                        else:
                            md_excerpt = "(TABLE Chatbot.md not available)"

                        # include the original raw LLM output if available to help the model understand format
                        original_raw = ''
                        try:
                            original_raw = resp.get('raw') or resp.get('content') or ''
                        except Exception:
                            original_raw = ''

                        # Append a machine-readable JSON schema excerpt to the retry prompt to force
                        # the model to use exact table/column names. We limit the size to avoid huge prompts.
                        try:
                            import json as _json
                            snap_for_json = get_schema_snapshot(limit_sample=1)
                            schema_obj = {t: info.get('columns', []) for t, info in snap_for_json.get('tables', {}).items()}
                            json_schema = _json.dumps(schema_obj, ensure_ascii=False, indent=2)
                            if len(json_schema) > 10000:
                                json_schema = json_schema[:10000] + "\n...\n"
                        except Exception:
                            json_schema = "(schema JSON not available)"

                        # Add concrete correction examples that map invented aliases to explicit SQL expressions
                        correction_examples = (
                            "Examples of corrective substitutions:\n"
                            "- purchase_frequency -> COUNT(DISTINCT Date) AS purchase_frequency\n"
                            "- months_active -> COUNT(DISTINCT substr(Date,1,7)) AS months_active\n"
                            "- total_qty -> SUM(CAST(REPLACE(COALESCE(NULLIF(Qte_Vendu, ''), '0'), ',', '.') AS NUMERIC)) AS total_qty\n"
                        )

                        retry_note = (
                            f"Previous SQL failed validation because it referenced unknown identifiers: {unknown_identifiers}."
                            " Please regenerate a corrected SQL using ONLY the exact table and column names listed in the AVAILABLE TABLES block."
                            " Return the JSON object with keys reasoning, sql_query, params (no extra text)."
                            "\n\nThe failing SQL was:\n" f"{original_sql}\n\n"
                            "The model reasoning provided previously was:\n" f"{original_reasoning}\n\n"
                            "The original raw model output was:\n" f"{original_raw}\n\n"
                            f"{correction_examples}"
                            "Refer to this PROJECT TABLE + BUSINESS RULES content to select exact column names and indicators:\n"
                            f"{md_excerpt}\n\n"
                            "MACHINE-READABLE SCHEMA (partial):\n"
                            f"{json_schema}"
                        )
                        # Append the retry note to the original question so the prompt includes context
                        retry_question = f"{question}\n\n{retry_note}"

                        resp_retry = llm_sql_agent.generate_sql_from_question(retry_question, sample_rows=[])
                        # timestamp and record the attempt for debugging
                        ts = datetime.datetime.utcnow().isoformat() + 'Z'
                        attempt_record = {
                            'attempt': attempt + 1,
                            'timestamp': ts,
                            'response': resp_retry,
                        }
                        retry_attempts.append(attempt_record)
                        logger.debug("LLM retry attempt %d at %s: %s", attempt + 1, ts, resp_retry)

                        # extract candidate SQL and reasoning from the model's response
                        candidate_sql = (resp_retry.get("sql_query") or "").strip()
                        candidate_sql = self._clean_sql(candidate_sql)
                        candidate_reasoning = resp_retry.get('reasoning') or resp_retry.get('explanation') or ''
                        # enrich the last attempt record with parsed candidate info
                        retry_attempts[-1].update({'candidate_sql': candidate_sql, 'candidate_reasoning': candidate_reasoning})

                        # Quick validation of the candidate SQL
                        try:
                            candidate_result = self.sql_validator.validate(candidate_sql)
                        except Exception:
                            candidate_result = None

                        # If SQL validation fails, continue to next attempt
                        if candidate_result is None or not getattr(candidate_result, 'is_valid', False):
                            # log invalid candidate result and continue
                            logger.debug("Candidate SQL failed sql_validator on attempt %d", attempt + 1)
                            continue

                        # Check identifiers again against known_tables
                        cand_identifiers = _extract_identifiers_from_sql(candidate_result.normalized_sql or candidate_sql)
                        cand_unknown = []
                        for ident in cand_identifiers:
                            low = ident.lower()
                            if low in known_tables:
                                continue
                            if any(low in cols for cols in known_tables.values()):
                                continue
                            cand_unknown.append(ident)

                        if not cand_unknown:
                            # success: use this SQL and proceed
                            fixed_sql = candidate_result.normalized_sql or candidate_sql
                            fixed_params = resp_retry.get('params') or {}
                            fixed_reasoning = candidate_reasoning or ''
                            break
                        else:
                            # prepare for next retry with updated unknown list
                            unknown_identifiers = cand_unknown
                            logger.debug("Candidate still references unknown identifiers on attempt %d: %s", attempt + 1, cand_unknown)

                    # If we obtained a fixed SQL, replace sql_text and sql_params and continue
                    if fixed_sql:
                        sql_text = fixed_sql
                        sql_params = fixed_params or {}
                        explanation = fixed_reasoning or explanation
                    else:
                        # retries exhausted; return structured error including attempts
                        full_msg = (
                            f"Generated SQL references unknown identifiers after {max_retries} retry attempts: {unknown_identifiers}. "
                            "LLM attempted regenerations are included in `llm_attempts`."
                        )
                        logger.error(full_msg)
                        return {
                            "success": False,
                            "validation_stage": "output_validation",
                            "error": full_msg,
                            "generated_sql": sql_text,
                            "unknown_identifiers": unknown_identifiers,
                            "llm_attempts": retry_attempts,
                        }
                except Exception:
                    # If something unexpected happened while retrying, return original message
                    logger.exception("LLM-driven retry for unknown identifiers failed")
                    return {
                        "success": False,
                        "validation_stage": "output_validation",
                        "error": msg + " (retry mechanism failed)",
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
                err_text = str(sqe)
                logger.error("SQLite OperationalError while executing SQL: %s", err_text)
                msg = f"SQLite execution error: {err_text}. Generated SQL: {mapped_sql}"
                return {
                    "success": False,
                    "validation_stage": "execution",
                    "error": msg,
                    "generated_sql": mapped_sql,
                    "params": normalized_params,
                    "sqlite_error": err_text,
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

    def _intent_sql_fallback(self, question: str) -> str:
        """Intent-aware local SQL fallback for a small set of common queries.

        Returns a SQL string (using physical table names like t_ventes_cleann)
        or empty string if no intent matched.
        """
        if not question:
            return ""
        q = question.lower()

        # Most loyal client: interpret as client with most months with purchases
        if 'most loyal' in q or 'most loyal client' in q or 'most loyal customers' in q:
            # months active (YYYY-MM) and total quantity as tiebreaker
            sql = (
                "SELECT Code_Client, Intitule_client, "
                "COUNT(DISTINCT substr(Date,1,7)) AS months_active, "
                "SUM(CAST(REPLACE(COALESCE(NULLIF(Qte_Vendu, ''), '0'), ',', '.') AS NUMERIC)) AS total_qty "
                "FROM t_ventes_cleann "
                "GROUP BY Code_Client, Intitule_client "
                "ORDER BY months_active DESC, total_qty DESC LIMIT 1"
            )
            return sql

        # Top N by sales
        m = re.search(r"top\s+(\d+)\s+by\s+sales", q)
        if m:
            n = int(m.group(1))
            sql = (
                "SELECT Ref_Article, Designation, "
                "SUM(CAST(REPLACE(COALESCE(NULLIF(Qte_Vendu, ''), '0'), ',', '.') AS NUMERIC)) AS qty_sold, "
                "SUM(CAST(REPLACE(COALESCE(NULLIF(CA_HT_NET, ''), '0'), ',', '.') AS NUMERIC)) AS net_sales "
                "FROM t_ventes_cleann "
                "GROUP BY Ref_Article, Designation ORDER BY net_sales DESC LIMIT %d"
            ) % n
            return sql

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
        # Additionally, detect ISO date string literals like '2025-09-16' and
        # replace them with named parameters to avoid inline literals in SQL.
        # This handles common patterns like "date >= '2025-09-16'" or BETWEEN clauses.
        date_pattern = re.compile(r"'(\d{4}-\d{2}-\d{2})'")

        def date_repl(m):
            date_val = m.group(1)
            pname = 'param_date'
            i = 1
            base = pname
            while pname in params:
                i += 1
                pname = f"{base}_{i}"
            params[pname] = date_val
            return f":{pname}"

        new_sql = date_pattern.sub(date_repl, new_sql)
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
    # Prefer the new text2sql QueryAgent if available for a unified backend.
    try:
        from text2sql.agent import QueryAgent as NewQueryAgent

        agent = NewQueryAgent()
        res = agent.generate_and_run(question)
        # Map to legacy response shape expected by older callers/tests
        rows = res.get("rows", [])
        sql = res.get("sql")
        return {
            "success": True,
            "validation_stage": "completed",
            "sql": sql,
            "original_sql": sql,
            "message": "SQL generated, validated and executed (via text2sql)",
            "params": {},
            "explanation": "",
            "columns": list(rows[0].keys()) if rows else [],
            "rows_preview": rows[:10],
            "rowcount": len(rows),
        }
    except Exception:
        # Fallback to legacy QueryAgent implementation in this module
        agent = QueryAgent()
        return agent.generate_sql(question)


# Alias for backward compatibility
generate_sql = generate_sql_from_question
    
