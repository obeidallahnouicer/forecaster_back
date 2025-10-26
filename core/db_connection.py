"""
Lightweight DB connection helpers for Text-to-SQL prototype.

Provides an in-memory SQLite database and utilities to load CSV/XLSX
files found in the repository root into tables so queries can be executed
without external dependencies.
"""
import sqlite3
from typing import Optional, List, Dict, Any
from pathlib import Path
import pandas as pd
import threading
import logging
import re

logger = logging.getLogger("core.db_connection")

# Global connection and lock
_conn: Optional[sqlite3.Connection] = None
_lock = threading.RLock()


def _sanitize_table_name(name: str) -> str:
    # Replace any non-alphanumeric character with underscore and prefix with t_
    import re
    stem = Path(name).stem
    safe = re.sub(r'[^0-9a-zA-Z]+', '_', stem)
    return "t_" + safe.lower()


def get_connection(load_files: bool = True) -> sqlite3.Connection:
    """Get or create a global in-memory sqlite3 connection.

    If load_files is True the function will attempt to load common CSV/XLSX
    files that exist in the workspace root (e.g. ventes_cleann.csv, STOCK.xlsx).
    """
    global _conn
    with _lock:
        if _conn is None:
            _conn = sqlite3.connect(":memory:")
            _conn.row_factory = sqlite3.Row
            logger.info("Created in-memory SQLite connection for Text-to-SQL prototype")

        if load_files:
            _load_workbook_files(_conn)

        return _conn


def _load_workbook_files(conn: sqlite3.Connection) -> None:
    """Load CSV/XLSX files into in-memory sqlite tables if not already present."""
    root = Path(".")
    # Only load files that are relevant to the application: STOCK and ventes_cleann
    candidates = [p for p in root.iterdir() if p.suffix.lower() in {".csv", ".xlsx", ".xls"}]
    # Filter to include files that are relevant to the application (stock, ventes_cleann, base)
    candidates = [p for p in candidates if any(k in p.stem.lower() for k in ("stock", "ventes_cleann", "base"))]

    for p in candidates:
        table = _sanitize_table_name(p.name)
        # If table exists, skip
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
        if cur.fetchone():
            continue

        try:
            if p.suffix.lower() == ".csv":
                # Try to auto-detect delimiter for CSVs (comma, semicolon, tab)
                # and try several encodings if the default fails (common on Windows/Excel exports).
                def _try_read_csv_with_encodings(path: Path):
                    encodings = ["utf-8-sig", "utf-8", "latin1", "cp1252"]
                    # first attempt: sniff delimiter using a small sample with utf-8-sig
                    import csv as _csv
                    sample = None
                    for enc in encodings:
                        try:
                            with path.open('r', encoding=enc, errors='replace') as fh:
                                sample = fh.read(8192)
                            if sample:
                                break
                        except Exception:
                            sample = None
                    sep = ','
                    try:
                        if sample:
                            dialect = _csv.Sniffer().sniff(sample)
                            sep = dialect.delimiter
                    except Exception:
                        # keep default comma
                        sep = ','

                    last_exc = None
                    for enc in encodings:
                        try:
                            df = pd.read_csv(path, sep=sep, engine='python', encoding=enc, on_bad_lines='skip')
                            logger.info(f"Read CSV {path.name} using encoding {enc} and separator '{sep}'")
                            return df
                        except Exception as e:
                            last_exc = e
                            logger.debug(f"Failed to read {path.name} with encoding {enc}: {e}")
                            continue
                    # final fallback: try pandas without specifying sep/encoding
                    try:
                        df = pd.read_csv(path, on_bad_lines='skip')
                        logger.info(f"Read CSV {path.name} with pandas default encoding")
                        return df
                    except Exception:
                        # raise the last encoding error to be caught by outer exception
                        raise last_exc or Exception("Failed to read CSV file")

                df = _try_read_csv_with_encodings(p)
            else:
                # read first sheet
                df = pd.read_excel(p, engine="openpyxl")

            # Normalize column names
            df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
            # Normalize date-like columns to ISO format (YYYY-MM-DD) so SQL ORDER BY works correctly
            # Detect common date column names (e.g., Date, date, Date_Vente) case-insensitive
            for col in list(df.columns):
                try:
                    if 'date' in col.lower():
                        # Parse many common date formats, prefer day-first (dd/mm/YYYY)
                        parsed = pd.to_datetime(df[col].astype(str), dayfirst=True, errors='coerce')
                        # If parsing succeeded for at least some rows, overwrite with ISO strings
                        if parsed.notna().sum() > 0:
                            df[col] = parsed.dt.strftime('%Y-%m-%d')
                            logger.info(f"Normalized date column '{col}' to ISO format")
                except Exception:
                    # Don't fail the whole load if date normalization fails
                    logger.debug(f"Date normalization failed for column {col}", exc_info=True)
            df.to_sql(table, conn, index=False)
            logger.info(f"Loaded {p.name} into sqlite table {table} ({len(df)} rows)")
        except Exception as e:
            logger.exception(f"Failed to load {p}: {e}")


def execute_select(sql: str, params: Optional[Dict[str, Any]] = None, max_rows: int = 1000) -> Dict[str, Any]:
    """Execute SELECT SQL against the in-memory connection and return results.

    Returns a dict with keys: columns, rows (list of dict), rowcount
    """
    conn = get_connection(load_files=True)
    params = params or {}
    cur = conn.cursor()
    # Ensure LIMIT protection isn't exceeded by caller; caller validator should add LIMIT
    try:
        # Quick sanitization: remove stray leading 't.' before table names (e.g., 't.t_stock')
        # which some LLM outputs may produce (interpreted incorrectly by SQLite as schema.table)
        try:
            sql = re.sub(r"\bt\.(t_[A-Za-z0-9_]+)\.", r"\1.", sql)
            sql = re.sub(r"\bt\.(t_[A-Za-z0-9_]+)\b", r"\1", sql)
        except Exception:
            pass

        cur.execute(sql, params)
        rows = cur.fetchmany(max_rows)
        cols = [d[0] for d in cur.description] if cur.description else []
        results = [dict(zip(cols, r)) for r in rows]
        # Count remaining rows if needed (cheap approximate)
        return {"columns": cols, "rows": results, "rowcount": len(results)}
    except Exception as e:
        # Include the SQL (truncated) in the logged message to help debugging
        try:
            snippet = (sql[:1000] + '...') if sql and len(sql) > 1000 else (sql or '')
        except Exception:
            snippet = '<unavailable>'
        logger.exception(f"SQL execution failed: {e}\nSQL: %s", snippet)
        # Re-raise a more informative OperationalError so callers can catch specifically
        import sqlite3 as _sqlite
        raise _sqlite.OperationalError(f"{e}; SQL: {snippet}")
