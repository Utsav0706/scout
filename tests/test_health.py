import pytest
from fastapi.testclient import TestClient

from scout import __version__


def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "db": "ok", "version": __version__}


def test_health_503_when_db_down(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scout.api.main.db_is_up", lambda _engine: False)
    resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json() == {"status": "degraded", "db": "down", "version": __version__}


def test_request_id_is_returned(client: TestClient) -> None:
    resp = client.get("/health", headers={"x-request-id": "abc123"})
    assert resp.headers["x-request-id"] == "abc123"
    assert len(client.get("/health").headers["x-request-id"]) == 32
