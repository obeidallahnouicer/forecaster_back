"""
Schema loader: inspects in-memory sqlite tables (or CSV/XLSX) and returns
compact schema descriptions (table, columns, sample values) to include in
LLM prompts.
"""
from typing import Dict, List, Any
from core.db_connection import get_connection
import sqlite3
import logging

logger = logging.getLogger("core.schema_loader")


def get_schema_snapshot(limit_sample: int = 5) -> Dict[str, Any]:
    """Return a compact schema snapshot for all tables in the in-memory DB.

    Output format:
    {"tables": {table_name: {"columns": [name,...], "samples": [{col:val}]}}}
    """
    conn = get_connection(load_files=True)
    cur = conn.cursor()
    tables = {}

    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    for (tname,) in cur.fetchall():
        try:
            ccur = conn.execute(f"SELECT * FROM {tname} LIMIT {limit_sample}")
            cols = [d[0] for d in ccur.description]
            rows = ccur.fetchall()
            samples = []
            for r in rows:
                samples.append({cols[i]: r[i] for i in range(len(cols))})

            tables[tname] = {"columns": cols, "samples": samples}
        except Exception as e:
            logger.exception(f"Failed to inspect table {tname}: {e}")
            continue

    return {"tables": tables}
