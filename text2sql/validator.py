import re
from typing import Dict, List, Tuple
from .logger import logger
try:
    import sqlparse
except Exception:  # pragma: no cover - optional dependency
    sqlparse = None

try:
    from rapidfuzz import process
except Exception:  # pragma: no cover - optional dependency
    process = None


IDENT_REGEX = re.compile(r"\b[\w\.\"]+\b")


def extract_identifiers(sql: str) -> List[str]:
    """Naive identifier extractor. For robust parsing use sqlparse if available."""
    if sqlparse:
        parsed = sqlparse.parse(sql)
        # Fallback simple implementation: return tokens that look like identifiers
    return list(set(re.findall(r"\"?([A-Za-z0-9_]+)\"?", sql)))


def is_select_only(sql: str) -> bool:
    s = sql.strip().lower()
    # crude check: disallow statements starting with insert/update/delete/alter/drop/create
    forbidden = ("insert", "update", "delete", "alter", "drop", "create", "truncate", "grant")
    return not any(s.startswith(f) for f in forbidden)


def fuzzy_fix_identifier(name: str, choices: List[str]) -> Tuple[str, float]:
    """Attempt to fuzzy-match name to the available choices.

    Returns (best_match, score) or (name, 0.0) if not found.
    """
    if process:
        best = process.extractOne(name, choices)
        if best:
            match, score, _ = best
            return match, score
    # fallback: exact substring match
    lowered = name.lower()
    for c in choices:
        if lowered == c.lower():
            return c, 100.0
    for c in choices:
        if lowered in c.lower() or c.lower() in lowered:
            return c, 80.0
    return name, 0.0


def validate_and_autocorrect(sql: str, schema: Dict[str, List[str]], fuzzy_threshold: float = 70.0) -> Tuple[str, List[str]]:
    """Validate SQL identifiers against schema and attempt autocorrections.

    Returns (corrected_sql, issues)
    """
    issues = []
    if not is_select_only(sql):
        issues.append("Non-SELECT or forbidden SQL detected")
        logger.warning("Validator rejected SQL: non-select detected")
        return sql, issues

    idents = extract_identifiers(sql)
    table_names = list(schema.keys())
    col_map = {t: set(c for c in cols) for t, cols in schema.items()}

    corrected_sql = sql

    # Try to replace unknown table/column names using fuzzy matching
    for ident in idents:
        # skip common SQL keywords
        if ident.lower() in ("select", "from", "where", "and", "or", "as", "on", "join", "count", "sum", "avg", "min", "max", "group", "order", "by", "limit"):
            continue

        # Check if ident is a table
        if ident in table_names:
            continue

        # Check if it's a column in any table
        found_col = False
        for t, cols in col_map.items():
            if ident in cols:
                found_col = True
                break

        if found_col:
            continue

        # Attempt fuzzy match against tables and columns
        best_table, table_score = fuzzy_fix_identifier(ident, table_names)
        # build flat list of columns prefixed with table for better suggestions
        flat_cols = []
        for t, cols in schema.items():
            flat_cols.extend(cols)
        best_col, col_score = fuzzy_fix_identifier(ident, flat_cols)

        if col_score >= fuzzy_threshold:
            logger.info("Auto-correcting identifier %s -> %s (score=%s)", ident, best_col, col_score)
            corrected_sql = re.sub(rf"\b{re.escape(ident)}\b", best_col, corrected_sql)
            issues.append(f"Autocorrected column {ident} -> {best_col} (score={col_score})")
        elif table_score >= fuzzy_threshold:
            logger.info("Auto-correcting identifier %s -> %s (score=%s)", ident, best_table, table_score)
            corrected_sql = re.sub(rf"\b{re.escape(ident)}\b", best_table, corrected_sql)
            issues.append(f"Autocorrected table {ident} -> {best_table} (score={table_score})")
        else:
            issues.append(f"Unknown identifier: {ident}")

    return corrected_sql, issues
