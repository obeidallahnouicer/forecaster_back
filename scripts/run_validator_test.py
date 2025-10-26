import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from guardrails.sql_validator import SQLOutputValidator

sql = (
    "SELECT Code_Client, Intitule_client, "
    "SUM(CAST(REPLACE(COALESCE(NULLIF(CA_HT_NET, ''), '0'), ',', '.') AS NUMERIC)) "
    "AS total_revenue FROM t_ventes_cleann GROUP BY Code_Client, Intitule_client ORDER BY total_revenue DESC LIMIT 10"
)

v = SQLOutputValidator(strict_mode=True)
res = v.validate(sql)
print('is_valid:', res.is_valid)
print('message:', res.message)
print('failure_reason:', res.failure_reason)
print('errors:', res.errors)
print('\nnormalized_sql:\n', res.normalized_sql)
