"""tools.rag.rag_sql_validator

Lightweight SQL validation utilities to enforce safe SQL output and formatting
constraints required by the system.
"""
from __future__ import annotations

import re
from typing import Tuple


_DANGEROUS_RE = re.compile(r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|EXEC|MERGE)\b", re.IGNORECASE)
_SELECT_OR_WITH = re.compile(r"^\s*(WITH\b|SELECT\b)", re.IGNORECASE)


def is_safe_sql(sql: str, allow_statements: Tuple[str, ...] = ("SELECT", "WITH")) -> bool:
    if sql is None:
        return False
    sql_nocomments = re.sub(r"--.*?$|/\*.*?\*/", "", sql, flags=re.DOTALL | re.MULTILINE)
    if _DANGEROUS_RE.search(sql_nocomments):
        return False
    if ";" in sql_nocomments.strip():
        return False
    if not _SELECT_OR_WITH.match(sql_nocomments):
        return False
    return True


def enforce_sql_format(sql: str) -> Tuple[bool, str]:
    if sql is None:
        return False, ""
    s = sql.strip()
    if s.startswith("```"):
        s = s.splitlines()[1:]
        s = "\n".join(s)
    if s.endswith(";"):
        s = s[:-1].rstrip()
    # Prefer to use the newer text2sql validator if available for autocorrection
    try:
        from text2sql.validator import validate_and_autocorrect

        corrected, issues = validate_and_autocorrect(s, {})
        # If issues contains 'Non-SELECT' then reject
        ok = not any(i.startswith("Non-SELECT") for i in issues)
        return ok, corrected
    except Exception:
        ok = is_safe_sql(s)
        return ok, s
