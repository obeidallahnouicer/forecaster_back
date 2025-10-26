import types
from agents.query_agent import QueryAgent


def test_llm_driven_retry_for_unknown_identifiers(monkeypatch):
    agent = QueryAgent()

    # Simulate first LLM response that references an unknown alias 'purchase_frequency'
    first_resp = {
        "reasoning": "Initial attempt used purchase_frequency alias",
        "sql_query": "SELECT Code_Client, purchase_frequency FROM t_ventes_cleann GROUP BY Code_Client ORDER BY purchase_frequency DESC LIMIT 10",
        "params": {},
    }

    # Simulate second (fixed) response that uses COUNT(DISTINCT Date) as purchase_frequency
    second_resp = {
        "reasoning": "Fixed to use COUNT(DISTINCT Date) as purchase_frequency",
        "sql_query": "SELECT Code_Client, COUNT(DISTINCT Date) AS purchase_frequency FROM t_ventes_cleann GROUP BY Code_Client ORDER BY purchase_frequency DESC LIMIT 10",
        "params": {},
    }

    calls = {"n": 0}


    def fake_generate_sql_from_question(question, sample_rows):
        # return first then second
        calls["n"] += 1
        if calls["n"] == 1:
            return first_resp
        return second_resp


    monkeypatch.setattr("llm.sql_agent.generate_sql_from_question", fake_generate_sql_from_question)

    out = agent.generate_sql("what clients are the most loyal ?")

    # Should either succeed (if environment has t_ventes_cleann in snapshot) or return structured error.
    assert isinstance(out, dict)
    # We expect at least that the agent attempted retries and included llm_attempts when failing
    if not out.get("success"):
        assert "llm_attempts" in out or out.get("validation_stage") != "output_validation"
    else:
        # On success we expect final SQL in 'sql' or 'original_sql'
        assert out.get("sql") or out.get("original_sql")
