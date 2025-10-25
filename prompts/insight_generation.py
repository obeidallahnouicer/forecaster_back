"""
Insight Generation Prompts

LangChain prompt templates for analyzing query results and generating insights.
"""

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
import os
import csv
from pathlib import Path
from typing import Optional
import time

# Optional pandas import - we'll try to use it for nicer inference but fall back to csv/openpyxl
try:
    import pandas as pd
except Exception:
    pd = None

INSIGHT_SYSTEM_PROMPT = """You are a business intelligence analyst specializing in data-driven insights.

YOUR ROLE:
- Analyze SQL query results and provide actionable business insights
- Identify trends, patterns, and anomalies
- Generate specific, measurable recommendations
- Use clear, concise business language

GUIDELINES:
- Focus on what the data reveals, not what it doesn't
- Quantify findings with specific numbers and percentages
- Prioritize actionable insights over generic observations
- Keep responses concise (2-3 sentences max)
- Avoid speculation or assumptions beyond the data
"""

INSIGHT_GENERATION_TEMPLATE = """Analyze the following query results and provide business insights.

CONTEXT:
User Question: {question}
SQL Query: {sql}

DATA SUMMARY:
- Total Rows: {row_count}
- Columns: {columns}
- Numeric Statistics: {stats}

FILES AND COLUMNS:
Provide the designation for each file included in the workspace/dataset, list the columns for that file and a one-line description of what each column represents. Format clearly.
{file_schema}

KEY METRICS:
{key_metrics}

TOP SAMPLE ROWS:
{sample_rows}

DETERMINISTIC INSIGHTS:
{deterministic_insights}

Based on this data, provide a concise analysis in 2-3 sentences:
1. What does this data tell us about the business?
2. What are the most important findings?
3. What actions should be taken?

Keep it specific, actionable, and focused on the data shown."""

# LangChain PromptTemplate
INSIGHT_GENERATION_PROMPT = PromptTemplate(
    input_variables=[
        "question",
        "sql",
        "row_count",
        "columns",
        "stats",
        "file_schema",
        "key_metrics",
        "sample_rows",
        "deterministic_insights"
    ],
    template=INSIGHT_GENERATION_TEMPLATE
)

# ChatPromptTemplate for chat models
INSIGHT_GENERATION_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", INSIGHT_SYSTEM_PROMPT),
    ("human", INSIGHT_GENERATION_TEMPLATE)
])

# Recommendation generation prompt
RECOMMENDATION_TEMPLATE = """Based on the data analysis, generate specific recommendations.

DATA CONTEXT:
{data_summary}

INSIGHTS FOUND:
{insights}

Generate 2-4 actionable recommendations in bullet points. Each recommendation should be:
- Specific and measurable
- Directly tied to the data
- Immediately actionable

Format as: 
- 📊 [Recommendation 1]
- ✅ [Recommendation 2]
- ⚠️ [Recommendation 3] (if there's a warning/concern)"""

RECOMMENDATION_PROMPT = PromptTemplate(
    input_variables=["data_summary", "insights"],
    template=RECOMMENDATION_TEMPLATE
)


def generate_file_schema(root: Optional[str] = None,
                         max_files: int = 20,
                         max_preview_rows: int = 3,
                         only_files: Optional[list] = None) -> str:
    """Scan the workspace for CSV/XLS/XLSX files and produce a formatted string describing
    each file, its columns, and a one-line description per column.

    This function tries to use pandas if available for dtype inference, otherwise it
    falls back to reading headers with the csv module or extracting Excel headers.
    Returns a plain text description intended for insertion into prompts.
    """
    root_path = Path(root) if root else Path.cwd()
    # Default to include common data files unless an explicit list is provided
    default_names = ["stock", "ventes_cleann", "base", "BASE", "STOCK"]
    include_names = [n.lower() for n in only_files] if only_files else default_names

    patterns = ["**/*.csv", "**/*.xlsx", "**/*.xls"]
    files = []
    for p in patterns:
        files.extend(list(root_path.glob(p)))

    # Filter files by name/stem (case-insensitive)
    filtered = []
    for f in files:
        stem = f.stem.lower()
        name = f.name.lower()
        if not only_files:
            # If no explicit list, include any of the default names or all files
            if any(n in stem or n == name for n in include_names) or True:
                filtered.append(f)
        else:
            if any(n in stem or n == name for n in include_names):
                filtered.append(f)

    # Deduplicate and limit
    files = sorted(set(filtered), key=lambda p: str(p))[:max_files]

    def infer_desc(col_name: str, sample_vals) -> str:
        name = col_name.lower()
        # heuristic-based short descriptions
        if any(k in name for k in ("date", "day", "month", "year")):
            return "Date or time indicator"
        if any(k in name for k in ("id", "ident", "uuid", "code")):
            return "Identifier"
        if any(k in name for k in ("price", "amount", "total", "sales", "revenue", "cost", "ca")):
            return "Monetary amount / sales value"
        if any(k in name for k in ("qty", "quantity", "units")):
            return "Quantity / units sold"
        if any(k in name for k in ("product", "sku", "item", "ref", "designation")):
            return "Product identifier or name"
        if any(k in name for k in ("store", "shop", "location", "region")):
            return "Store or location"
        # fallback using sample values
        if sample_vals is not None and len(sample_vals) > 0:
            s = sample_vals[0]
            try:
                float(str(s))
                return "Numeric value"
            except Exception:
                pass
        return "Free-text / categorical"

    out_lines = []
    if not files:
        return "No CSV/XLS/XLSX data files found in workspace."

    for f in files:
        try:
            out_lines.append(f"- File: {f.name}")
            # read header and sample values
            cols = []
            samples = {}
            if pd is not None and f.suffix.lower() in (".csv", ".xlsx", ".xls"):
                try:
                    if f.suffix.lower() == ".csv":
                        df = pd.read_csv(f, nrows=max_preview_rows, on_bad_lines='skip')
                    else:
                        df = pd.read_excel(f, nrows=max_preview_rows)
                    cols = list(df.columns)
                    for c in cols:
                        try:
                            samples[c] = df[c].dropna().astype(str).tolist()[:max_preview_rows]
                        except Exception:
                            samples[c] = []
                except Exception:
                    cols = []
            if not cols:
                # fallback: read header line for CSV
                if f.suffix.lower() == ".csv":
                    try:
                        with f.open("r", encoding="utf-8", errors="ignore") as fh:
                            reader = csv.reader(fh)
                            header = next(reader, [])
                            cols = header
                    except Exception:
                        cols = []
                elif f.suffix.lower() in (".xlsx", ".xls"):
                    # Without pandas we can't read xlsx reliably
                    cols = ["(unable to read columns without pandas)"]

            # If pandas returned a single header string containing semicolons, split it
            if cols and len(cols) == 1 and isinstance(cols[0], str) and ";" in cols[0]:
                cols = [c.strip() for c in cols[0].split(";") if c.strip()]

            out_lines.append("  - Columns:")
            for c in cols:
                sample_vals = samples.get(c, [])
                desc = infer_desc(c, sample_vals)
                out_lines.append(f"    - {c} \u2014 {desc}")
            out_lines.append("")
        except Exception as exc:
            out_lines.append(f"- File: {f.name} \u2014 error reading file: {exc}")
            out_lines.append("")

    return "\n".join(out_lines)


# Simple in-memory cache for file_schema with TTL
_CACHED_SCHEMA = None
_CACHED_SCHEMA_TS = 0
_CACHED_SCHEMA_TTL = 300  # seconds


def get_cached_file_schema(force_refresh: bool = False, ttl: Optional[int] = None) -> str:
    """Return cached file schema, refreshing if TTL expired or force_refresh=True."""
    global _CACHED_SCHEMA, _CACHED_SCHEMA_TS, _CACHED_SCHEMA_TTL
    if ttl is not None:
        _CACHED_SCHEMA_TTL = ttl

    now = time.time()
    if force_refresh or _CACHED_SCHEMA is None or (now - _CACHED_SCHEMA_TS) > _CACHED_SCHEMA_TTL:
        _CACHED_SCHEMA = generate_file_schema()
        _CACHED_SCHEMA_TS = now

    return _CACHED_SCHEMA

