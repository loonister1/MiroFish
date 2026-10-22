import pytest

from app.config import Config


@pytest.fixture(autouse=True)
def _no_local_graph_database(monkeypatch):
    # .env may point at the real local Neo4j; tests opt in explicitly.
    monkeypatch.setattr(Config, "NEO4J_URI", None, raising=False)
