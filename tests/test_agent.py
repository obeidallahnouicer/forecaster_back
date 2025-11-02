import os
import sqlite3
import pytest

from text2sql.model_loader import ModelLoader
from text2sql.agent import Chat2DBQueryAgent


@pytest.fixture(autouse=True)
def use_mock_model(monkeypatch):
    # Force the model loader to use the lightweight mock
    monkeypatch.setenv("MODEL_USE_MOCK", "1")
    yield


def test_generate_and_run_count(tmp_path):
    # Create an on-disk sqlite file so SQLAlchemy can inspect it
    db_path = tmp_path / "data.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE singer (Singer_ID INTEGER PRIMARY KEY, Name TEXT);")
    cur.executemany("INSERT INTO singer (Singer_ID, Name) VALUES (?, ?)", [(1, "A"), (2, "B"), (3, "C")])
    conn.commit()
    conn.close()

    db_url = f"sqlite:///{db_path}"
    agent = Chat2DBQueryAgent(db_url=db_url, model_loader=ModelLoader())
    res = agent.generate_and_run("How many singers do we have?")

    assert "rows" in res
    # mock returns SELECT COUNT(*) AS count FROM singer;
    assert res["rows"][0].get("count") == 3
