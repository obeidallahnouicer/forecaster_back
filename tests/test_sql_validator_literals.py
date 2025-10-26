import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from guardrails.sql_validator import SQLOutputValidator


def test_accepts_format_literals():
    sql = (
        "SELECT Code_Client, Intitule_client, "
        "SUM(CAST(REPLACE(COALESCE(NULLIF(CA_HT_NET, ''), '0'), ',', '.') AS NUMERIC)) "
        "AS total_revenue FROM t_ventes_cleann GROUP BY Code_Client, Intitule_client ORDER BY total_revenue DESC LIMIT 10"
    )
    v = SQLOutputValidator(strict_mode=True)
    res = v.validate(sql)
    assert res.is_valid


def test_rejects_user_string_literal():
    sql = "SELECT * FROM t_users WHERE name = 'Alice'"
    v = SQLOutputValidator(strict_mode=True)
    res = v.validate(sql)
    assert not res.is_valid
