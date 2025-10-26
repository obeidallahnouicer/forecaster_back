import json

from llm.sql_agent import parse_and_normalize_llm_response


def test_valid_new_shape():
    content = json.dumps({
        "reasoning": "Used t_ventes_cleann and aggregated Qte_Vendu",
        "sql_query": "SELECT Ref_Article, SUM(CAST(REPLACE(COALESCE(NULLIF(Qte_Vendu, ''), '0'), ',', '.') AS NUMERIC)) AS total FROM ventes_cleann GROUP BY Ref_Article",
        "params": {"year": 2025},
    })
    resp = {"content": content}
    out = parse_and_normalize_llm_response(content, resp)
    assert out["reasoning"].startswith("Used")
    assert out["sql_query"].strip().lower().startswith("select")
    assert out["params"]["year"] == 2025


def test_valid_old_shape():
    content = json.dumps({
        "sql": "SELECT 1 as one",
        "params": {},
        "explanation": "legacy",
    })
    resp = {"content": content}
    out = parse_and_normalize_llm_response(content, resp)
    assert out["reasoning"] == "legacy"
    assert out["sql_query"].strip().lower().startswith("select")


def test_invalid_json():
    content = "not a json"
    resp = {"content": content}
    out = parse_and_normalize_llm_response(content, resp)
    assert out["reasoning"] == "LLM did not return valid JSON"
    assert out["sql_query"] == ""


def test_non_select_sql_rejected():
    content = json.dumps({"reasoning": "ok", "sql_query": "DROP TABLE users;"})
    resp = {"content": content}
    out = parse_and_normalize_llm_response(content, resp)
    assert out["sql_query"] == ""
    assert "does not start with SELECT" in out["reasoning"]
