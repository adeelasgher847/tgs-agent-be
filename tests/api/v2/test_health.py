"""v2 health endpoint — public, no auth."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


def test_v2_health_returns_200_without_auth():
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/api/v2/health")

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "version": settings.APP_VERSION,
        "service": "fastapi-v2",
        "db": "ok",
    }
