"""SQL Generation prompt helpers

This module builds a strict Text-to-SQL prompt that includes the live DB schema
(physical table names and columns) so the LLM always knows which table contains
which columns. The prompt instructs the model to prefer physical tables like
`t_ventes_cleann` and `t_stock`, and to aggregate from sales tables when
logical helper tables (e.g., `t_clients`) are not present.
"""

from typing import List, Dict, Optional
try:
    from langchain_core.prompts import PromptTemplate
except Exception:
    # Minimal local fallback for PromptTemplate so the module can be imported
    # in environments where langchain_core is not installed (tests, CI).
    class PromptTemplate:
        def __init__(self, input_variables=None, template=""):
            self.input_variables = input_variables or []
            self.template = template

        def format(self, **kwargs):
            return self.template.format(**kwargs)
from core.schema_loader import get_schema_snapshot

# The only allowed columns for `ventes_cleann` as provided by the original design
VENTES_CLEANN_COLUMNS = [
    "Client_Principal", "Code_Client", "Intitule_client", "Categorie_Client",
    "Pays", "Zone", "Gouvernorat", "Adresse", "Representant", "NDocument",
    "Type_Document", "Date", "Mois", "Annee", "Marque", "Famille",
    "Sous_Famille", "Ref_Article", "Designation", "Qte_Vendu", "CA_HT_BRUT",
    "Tx_Remise", "CA_HT_NET",
]


def build_schema_text_from_snapshot(limit_sample: int = 1) -> str:
    """Build a machine-readable schema block from the current DB snapshot.

    This lists physical table names (e.g., `t_ventes_cleann`, `t_stock`) and their
    available columns. The LLM prompt will use this to decide which tables to query.
    """
    snap = get_schema_snapshot(limit_sample=limit_sample)
    lines: List[str] = []
    for tname, info in snap.get('tables', {}).items():
        cols = info.get('columns', [])
        lines.append(f"TABLE: {tname}")
        lines.append("- columns: " + ", ".join(cols))
        # include a tiny sample if available
        samples = info.get('samples') or []
        if samples:
            sample = samples[0]
            sample_items = ", ".join([f"{k}={repr(v)[:30]}" for k, v in list(sample.items())[:5]])
            lines.append("- example: " + sample_items)
    if not lines:
        return "(no tables available)"
    return "\n".join(lines)


def build_json_schema_from_snapshot(limit_sample: int = 1) -> str:
    """Return a compact JSON representation of the available tables and columns.

    This is used to give the LLM an exact machine-readable list that it must follow.
    """
    snap = get_schema_snapshot(limit_sample=limit_sample)
    j = {"tables": {}}
    for tname, info in snap.get('tables', {}).items():
        j["tables"][tname] = {"columns": info.get('columns', [])}
    import json
    return json.dumps(j, ensure_ascii=False, indent=2)


# Backwards-compatible system prompt variable (some modules import this name)
SQL_SYSTEM_PROMPT = (
    "You are an expert SQL query generator for the repository's in-memory SQLite database.\n"
    "CRITICAL: Under NO CIRCUMSTANCES invent or reference tables or columns that are NOT listed in the AVAILABLE TABLES block.\n"
    "Only use the physical table names and column names shown in the schema section. If the user asks about a logical\n"
    "entity (for example 'clients' or 'products') and no helper table (e.g., t_clients) exists, compute the requested\n"
    "metrics by aggregating from the available physical sales table(s) (for example `t_ventes_cleann`) using only the\n"
    "columns shown in the schema. Always explain when you derived values instead of using a dedicated helper table.\n"
    "Map sales/revenue to CA_HT_NET and quantities to Qte_Vendu.\n"
    "IMPORTANT: For any numeric columns that may be stored as text (commas, empty strings, nulls, or scientific notation),\n"
    "use SQL expressions that safely convert values, e.g. COALESCE(NULLIF(col, ''), '0'), REPLACE(col, ',', '.') and CAST(... AS NUMERIC).\n"
    "Return ONLY a single JSON object (no markdown, no commentary) with the EXACT keys: reasoning, sql_query, params.\n"
    "- `reasoning` (string): brief step-by-step reasoning describing which physical tables/columns and expressions you used.\n"
    "- `sql_query` (string): a single SELECT statement that uses only the physical table and column names from the AVAILABLE TABLES block.\n"
    "- `params` (object): an object of named parameter values (use named parameters like :since_date).\n"
    "If you cannot produce a valid SELECT using only the available tables/columns, return a JSON object with `sql_query` set to an empty string and put the explanation in `reasoning`.\n"
    "Do NOT include any text before or after the JSON object. The JSON object must start with '{' as the very first character of your response.\n"
    "The SQL must be valid SQLite and must NOT include multiple statements."
)


def build_schema_text(stock_columns: Optional[List[str]] = None) -> str:
    """Return a machine-readable schema text for the prompt (legacy/backup).

    This is kept for callers that supply sample rows or a custom stock columns list.
    """
    ventes_block = "TABLE: ventes_cleann\n- columns: " + ", ".join(VENTES_CLEANN_COLUMNS)
    stock_block = ""
    if stock_columns:
        stock_block = "\nTABLE: STOCK\n- columns: " + ", ".join(stock_columns)
    return ventes_block + stock_block


def build_sample_rows_text(sample_rows: List[Dict[str, object]]) -> str:
    """Produce a compact table text showing the first two sample rows.

    Expects list of dicts keyed by columns. Will include only the first two rows
    and only keys present in VENTES_CLEANN_COLUMNS.
    """
    rows = sample_rows[:2]
    if not rows:
        return "(no sample rows provided)"
    # build header from ventes columns intersection with row keys
    header = [c for c in VENTES_CLEANN_COLUMNS if c in rows[0].keys()]
    lines: List[str] = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "---|" * len(header))
    for r in rows:
        vals = [str(r.get(h, "")) for h in header]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


SQL_GENERATION_TEMPLATE = (
    "You are a Text-to-SQL assistant for a business intelligence system.\n\n"

    "RULES YOU MUST FOLLOW:\n"
    "1) USE ONLY the physical tables and columns listed in the AVAILABLE TABLES block. DO NOT invent or reference any other table or column.\n"
    "2) Prefer the physical sales table (for example `t_ventes_cleann`) for sales aggregations.\n"
    "3) If a logical helper table such as `t_clients` or `t_produits` is NOT present, DO NOT reference it. Instead, compute client/product\n"
    "   level metrics by aggregating from the available sales table(s) (for example group by Code_Client from `t_ventes_cleann`).\n"
    "   The SQL you output MUST use only columns listed in the AVAILABLE TABLES block. In the explanation, explicitly state which\n"
    "   physical table and which computed expressions you used to derive the metric.\n"
    "4) For sales/revenue use CA_HT_NET. For quantities use Qte_Vendu.\n"
    "5) For stock analysis use the physical stock table if present (e.g., `t_stock`) and its columns.\n"
    "6) Only generate SELECT queries. Use named parameters (:param_name) for user values.\n"
    "7) Group by categorical columns when aggregating and include ORDER/LIMIT for top-N queries.\n"
    "8) Apply business rules when relevant (inactive clients: Mois_Depuis_Derniere_Vente >= 3,\n"
    "   stock alerts: Couverture_Stock < 1 or > 6, sales decline: Variation_CA_% < -10).\n"
    "9) Return valid JSON with keys: reasoning, sql_query, params.\n\n"

    "AVAILABLE TABLES AND COLUMNS (physical names):\n{schema}\n\n"
    "MACHINE-READABLE SCHEMA (JSON) - MUST be used by the model to select exact column names:\n{json_schema}\n\n"

    "BUSINESS CONTEXT (for mapping user intent to metrics):\n"
    "- Clients inactifs: Mois_Depuis_Derniere_Vente >= 3 (or compute months since last sale from sales table)\n"
    "- Clients à risque: ΔCA_% < -15 AND Frequence_Achat_Moyenne < 2\n"
    "- Rupture stock: Couverture_Stock < 1\n"
    "- Surstock: Couverture_Stock > 6\n"
    "- Baisse ventes: Variation_CA_% < -10 OR Variation_Qte_% < -15\n"
    "- Forte croissance: ΔCA_% > +20\n\n"

    "USER QUESTION: {question}\n\n"

    "OUTPUT: Return ONLY valid JSON with this structure:\n"
    "- reasoning: brief explanation string describing your logic and which physical tables/expressions you used (short).\n"
    "- sql_query: a single SELECT statement string that uses only the physical table/column names shown above.\n"
    "- params: object with parameter values (use named parameters like :param_name).\n\n"

    "If the exact logical table requested by the user does not exist, compute the answer\n"
    "by aggregating from the appropriate physical tables and clearly document this in the explanation. DO NOT reference or create\n"
    "tables that are not in the AVAILABLE TABLES block. If a derived metric (for example months since last sale) is required,\n"
    "compute it from existing timestamp/date columns (for example `Date`) using SQLite expressions and include that expression in the SQL.\n"
    "\nEXAMPLES (few-shot):\n"
    "# Example 1 - user asks top seller by number of distinct clients in last 6 months\n"
    "# Available tables: t_ventes_cleann with columns [Code_Client, Representant, Date, CA_HT_NET, ...]\n"
    "# Correct JSON response (ONLY the JSON object; keys must be reasoning, sql_query, params):\n"
    "{{\n  \"reasoning\": \"Aggregated from t_ventes_cleann using exact column names; computed since_date as 6 months ago.\",\n"
    "  \"sql_query\": \"SELECT Representant, COUNT(DISTINCT Code_Client) AS nb_clients FROM t_ventes_cleann WHERE Date >= :since_date GROUP BY Representant ORDER BY nb_clients DESC LIMIT 10\",\n"
    "  \"params\": {{\"since_date\": \"2025-04-01\"}}\n}}\n"
    "# Example 2 - numeric-safe casting example when amounts may be text with commas\n"
    "# User: 'Show top products by revenue last month'\n"
    "# Correct JSON response demonstrating numeric-safe cast in SQL:\n"
    "{{\n  \"reasoning\": \"Summed CA_HT_NET after replacing comma decimals and casting to numeric; using t_ventes_cleann.\",\n"
    "  \"sql_query\": \"SELECT Ref_Article, SUM(CAST(REPLACE(COALESCE(NULLIF(CA_HT_NET, ''), '0'), ',', '.') AS NUMERIC)) AS total_revenue FROM t_ventes_cleann WHERE Date >= :since_date GROUP BY Ref_Article ORDER BY total_revenue DESC LIMIT 10\",\n"
    "  \"params\": {{\"since_date\": \"2025-09-01\"}}\n}}\n"
    "# Never reference tables that are not listed above (e.g., do NOT use t_clients).\n"
)


def build_sql_generation_prompt(question: str, sample_rows: List[Dict[str, object]], stock_columns: Optional[List[str]] = None) -> str:
    """Construct the full prompt text to send to the LLM.

    - Inserts the machine-readable schema (live snapshot preferred)
    - Inserts the first two sample rows
    - Enforces the rules from the user's instruction
    """
    # Prefer live snapshot so the prompt lists exact physical tables present in the DB
    try:
        schema_text = build_schema_text_from_snapshot(limit_sample=1)
    except Exception:
        # Fallback to legacy static schema text if snapshot fails
        schema_text = build_schema_text(stock_columns)
    samples_text = build_sample_rows_text(sample_rows)
    # For compatibility with other modules that expect a 'file_schema' field,
    # provide the same schema_text as file_schema as a machine-readable representation.
    # Also include a compact JSON schema used by the generation template.
    try:
        json_schema = build_json_schema_from_snapshot(limit_sample=1)
    except Exception:
        json_schema = "{}"

    prompt = SQL_GENERATION_TEMPLATE.format(
        schema=schema_text,
        file_schema=schema_text,
        json_schema=json_schema,
        sample_rows=samples_text,
        question=question,
    )
    # Append the sample rows block (first two) so callers that format the template
    # without sample_rows still receive the examples when using this helper.
    prompt += "\n\nSAMPLE ROWS (first two):\n" + samples_text + "\n"
    return prompt


# Backwards-compatible LangChain PromptTemplate (if callers expect one)
SQL_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "sample_rows", "question"],
    template=(
        "{question}\n\nSchema:\n{schema}\n\nSample rows:\n{sample_rows}\n\n"
        "Return a single JSON object with keys: sql, params, explanation."
    ),
)

__all__ = [
    "VENTES_CLEANN_COLUMNS",
    "build_schema_text",
    "build_sample_rows_text",
    "build_sql_generation_prompt",
    "SQL_GENERATION_PROMPT",
]
