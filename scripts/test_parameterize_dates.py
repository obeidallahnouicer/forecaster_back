import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.query_agent import QueryAgent

q = QueryAgent()
sql = "SELECT * FROM t_ventes_cleann WHERE Date >= '2025-09-16' AND Code_Client = 'ABC'"
new_sql, params = q._parameterize_intervals(sql, {})
print('new_sql:', new_sql)
print('params:', params)
