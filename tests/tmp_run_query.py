import sys, time, json
sys.path.append(r'c:\Users\onouicer\Desktop\slimback\forecaster_back')
from agents.query_agent import generate_sql
from core.db_connection import execute_select

question = 'which zone or governorat sold the most products ?'
start = time.time()
res = generate_sql(question)
print('LLM gen result:')
print(json.dumps(res, default=str, indent=2, ensure_ascii=False))

if res.get('success') and res.get('sql'):
    start = time.time()
    out = execute_select(res['sql'], params=None, max_rows=10)
    exec_time = (time.time()-start)*1000
    print('\nExecution result:')
    print(json.dumps({'columns': out['columns'], 'rows_preview': out['rows'], 'rowcount': out['rowcount'], 'execution_time_ms': exec_time}, default=str, indent=2, ensure_ascii=False))
else:
    print('\nNo executable SQL returned')
