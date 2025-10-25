from core.db_connection import get_connection

conn = get_connection(load_files=True)
cur = conn.cursor()
# Create derived clients table for compatibility with LLM-generated queries
try:
    cur.execute('''
    CREATE TABLE IF NOT EXISTS t_clients AS
    SELECT
      Code_Client,
      Intitule_client AS Intitule_Client,
      Representant,
      MAX(Date) AS Date_Derniere_Vente,
      CAST((julianday('now') - julianday(MAX(Date))) / 30 AS INTEGER) AS Mois_Depuis_Derniere_Vente,
      SUM(CA_HT_NET) AS CA_Total
    FROM t_ventes_cleann
    GROUP BY Code_Client, Intitule_client, Representant
    ''')
    conn.commit()
    print('t_clients created (or already present)')
except Exception as e:
    print('Failed to create t_clients:', e)

# Show sample rows
for row in conn.execute('SELECT Code_Client, Intitule_Client, Representant, Mois_Depuis_Derniere_Vente, CA_Total FROM t_clients LIMIT 5'):
    print(row)
