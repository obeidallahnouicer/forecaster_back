"""Business rule parser and SQL test-generator.

This module parses simple rule lines from the project's markdown files and
generates safe SELECT SQL queries that can be executed against sample tables
to validate business rules end-to-end.

The parser uses heuristics to map textual field names to canonical column
names and selects an appropriate test table (clients, stock, kpi_performance,
ventes_mensuelles). It's intentionally conservative: generated SQLs are
SELECT COUNT(*) queries used in tests to assert presence/absence of rows.
"""
import re
from typing import List, Dict, Any
from pathlib import Path


DEFAULT_COLUMN_MAP = {
    # textual -> canonical column name used in test DB
    "ΔCA_%": "Delta_CA_Percent",
    "ΔQte_%": "Delta_Qte_Percent",
    "Variation_CA_%": "Variation_CA_Percent",
    "Variation_CA_Percent": "Variation_CA_Percent",
    "CA_HT_NET": "CA_HT_NET",
    "Mois_Depuis_Derniere_Vente": "Mois_Depuis_Derniere_Vente",
    "Couverture_Stock": "Couverture_Stock",
    "CA_Moyen_Annuel": "CA_Moyen_Annuel",
}


def normalize_field_name(name: str) -> str:
    s = name.strip()
    # replace unicode delta with 'Delta_'
    s = s.replace("Δ", "Delta_")
    s = s.replace("%", "Percent")
    s = re.sub(r"[^0-9A-Za-z_]+", "_", s)
    s = re.sub(r"__+", "_", s)
    return s.strip("_")


def guess_table_for_field(field: str) -> str:
    f = field.lower()
    if "stock" in f or "couverture" in f:
        return "stock"
    if "client" in f or "mois" in f or "ca_moyen" in f or "ca_ht" in f:
        return "clients"
    # default to kpi_performance for percentage KPIs
    if "percent" in f or "delta" in f or "variation" in f:
        return "kpi_performance"
    return "ventes_mensuelles"


def parse_rules(md_path: str) -> List[Dict[str, Any]]:
    """Parse the markdown file and return a list of rule dicts.

    Each rule dict contains: original_text, field, op, value, suggested_table.
    """
    p = Path(md_path)
    if not p.exists():
        raise FileNotFoundError(md_path)

    txt = p.read_text(encoding="utf-8")

    # Find lines that look like rules: contain a comparator and a number
    pattern = re.compile(r"([A-Za-z0-9_\u00B0\u00C0-\u017F%Δ\-\s\\]+?)\s*(<|>|<=|>=|=)\s*([+\-]?\d+)%?", flags=re.I)
    rules = []
    for m in pattern.finditer(txt):
        raw_field = m.group(1).strip()
        op = m.group(2)
        val = int(m.group(3))
        field_norm = normalize_field_name(raw_field)
        # map to canonical column if available
        col = DEFAULT_COLUMN_MAP.get(raw_field, DEFAULT_COLUMN_MAP.get(field_norm, field_norm))
        table = guess_table_for_field(col)
        rules.append({
            "text": m.group(0).strip(),
            "raw_field": raw_field,
            "field": col,
            "op": op,
            "value": val,
            "table": table,
        })

    return rules


def rule_to_sql(rule: Dict[str, Any]) -> str:
    """Generate a safe SELECT COUNT(*) SQL for the rule.

    Example: {field: 'Delta_CA_Percent', op: '<', value: -10, table: 'kpi_performance'}
    -> SELECT COUNT(*) AS cnt FROM kpi_performance WHERE Delta_CA_Percent < -10;
    """
    table = rule["table"]
    field = rule["field"]
    op = rule["op"]
    value = rule["value"]

    # For percentages the rule values in md are often negative when describing drops
    sql = f"SELECT COUNT(*) AS cnt FROM {table} WHERE {field} {op} {value};"
    return sql
