import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.query_agent import _extract_identifiers_from_sql

sql = "SELECT Code_Client, COUNT(DISTINCT Date) AS purchase_frequency FROM t_ventes_cleann GROUP BY Code_Client ORDER BY purchase_frequency DESC LIMIT 10"
ids = _extract_identifiers_from_sql(sql)
print('Extracted identifiers:', ids)
