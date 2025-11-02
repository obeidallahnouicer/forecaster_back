import os
import sqlite3
import tempfile
from text2sql.md_rules import load_example_questions
from text2sql.model_loader import ModelLoader
from text2sql.agent import Chat2DBQueryAgent


def setup_clients_db(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE clients (
            Code_Client TEXT PRIMARY KEY,
            Intitule_Client TEXT,
            CA_Moyen_Annuel REAL,
            Mois_Depuis_Derniere_Vente INTEGER,
            Variation_CA_Percent REAL
        );
        """
    )
    # Insert sample rows
    rows = [
        ("CL001", "AYADI HOUWAIDA", 10000.0, 1, -5.0),
        ("CL002", "CLIENT B", 8000.0, 4, -25.0),
        ("CL003", "CLIENT C", 12000.0, 6, -30.0),
        ("CL004", "CLIENT D", 5000.0, 0, 10.0),
    ]
    cur.executemany("INSERT INTO clients VALUES (?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()


def test_md_examples_execute(tmp_path, monkeypatch):
    # Force mock model
    monkeypatch.setenv("MODEL_USE_MOCK", "1")

    db_file = tmp_path / "clients.db"
    setup_clients_db(str(db_file))
    db_url = f"sqlite:///{db_file}"

    # Load questions from the project's TABLE Chatbot.md
    md_path = os.path.join(os.path.dirname(__file__), "..", "TABLE Chatbot.md")
    md_path = os.path.abspath(md_path)
    questions = load_example_questions(md_path)

    agent = Chat2DBQueryAgent(db_url=db_url, model_loader=ModelLoader())

    assert len(questions) > 0, "No example questions found in TABLE Chatbot.md"

    # Run a subset of questions that are client-related
    client_qs = [q for q in questions if "client" in q.lower() or "clients" in q.lower()]
    assert client_qs, "No client-related example questions found"

    for q in client_qs:
        res = agent.generate_and_run(q)
        assert "sql" in res and res["sql"].strip().lower().startswith("select"), f"Non-select SQL generated for: {q}"
        assert "rows" in res and isinstance(res["rows"], list)
