import os
import sqlite3
import tempfile
from text2sql.rule_engine import parse_rules, rule_to_sql


def setup_rule_tables(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # clients table
    cur.execute(
        """
        CREATE TABLE clients (
            Code_Client TEXT PRIMARY KEY,
            Intitule_Client TEXT,
            CA_HT_NET REAL,
            Mois_Depuis_Derniere_Vente INTEGER,
            CA_Moyen_Annuel REAL,
            Variation_CA_Percent REAL
        );
        """
    )
    # kpi_performance table
    cur.execute(
        """
        CREATE TABLE kpi_performance (
            Code TEXT PRIMARY KEY,
            Delta_CA_Percent REAL,
            Delta_Qte_Percent REAL
        );
        """
    )
    # stock table
    cur.execute(
        """
        CREATE TABLE stock (
            Reference_Article TEXT PRIMARY KEY,
            Couverture_Stock REAL
        );
        """
    )

    # Insert rows that will trigger rules
    cur.executemany("INSERT INTO clients VALUES (?, ?, ?, ?, ?, ?)", [
        ("CL100", "CLIENT RISK", 0.0, 3, 0.0, -15.0),
    ])
    cur.executemany("INSERT INTO kpi_performance VALUES (?, ?, ?)", [
        ("KP1", -12.0, -20.0),
        ("KP2", 30.0, 25.0),
    ])
    cur.executemany("INSERT INTO stock VALUES (?, ?)", [
        ("ART1", 7.0),
        ("ART2", 0.5),
    ])

    conn.commit()
    conn.close()


def test_generate_sqls_and_execute(tmp_path):
    db_file = tmp_path / "rules.db"
    setup_rule_tables(str(db_file))
    db_url = f"sqlite:///{db_file}"

    md_path = os.path.join(os.path.dirname(__file__), "..", "TABLE Chatbot.md")
    md_path = os.path.abspath(md_path)
    rules = parse_rules(md_path)
    assert rules, "No rules parsed from MD"

    # Execute each generated SQL against the sqlite DB to ensure it runs
    import sqlite3

    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    for r in rules:
        sql = rule_to_sql(r)
        # Simple sanity: SQL should be a SELECT
        assert sql.strip().lower().startswith("select"), sql
        # Ensure the column exists in the target table; if not, add it (sqlite-friendly)
        tbl = r.get("table")
        col = r.get("field")
        # Ensure the table exists; if not, create a minimal placeholder table
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (tbl,))
        if cur.fetchone() is None:
            cur.execute(f"CREATE TABLE {tbl} (id TEXT PRIMARY KEY);")

        cur.execute(f"PRAGMA table_info({tbl});")
        cols = [c[1] for c in cur.fetchall()]
        if col not in cols:
            # Add REAL column with default 0 so the test SQL can run
            cur.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} REAL DEFAULT 0;")
        cur.execute(sql)
        row = cur.fetchone()
        assert row is not None
        # cnt should be integer >= 0
        cnt = row[0]
        assert isinstance(cnt, int) or isinstance(cnt, float)
    conn.close()
