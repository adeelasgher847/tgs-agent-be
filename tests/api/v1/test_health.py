"""v1 health endpoint — public, no auth."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


def test_v1_health_returns_200_without_auth():
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/api/v1/health")

    assert resp.status_code == 200


def test_v1_health_response_schema():
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/api/v1/health")

    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == settings.APP_VERSION
    assert "timestamp" in body
    assert "betterstack_badge" in body


def test_v1_health_betterstack_badge_defaults_none():
    prev = settings.BETTERSTACK_BADGE_URL
    settings.BETTERSTACK_BADGE_URL = None
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/api/v1/health")
        assert resp.json()["betterstack_badge"] is None
    finally:
        settings.BETTERSTACK_BADGE_URL = prev


def test_v1_health_betterstack_badge_reflects_setting():
    prev = settings.BETTERSTACK_BADGE_URL
    settings.BETTERSTACK_BADGE_URL = "https://status.yourdomain.com/badge"
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/api/v1/health")
        assert resp.json()["betterstack_badge"] == "https://status.yourdomain.com/badge"
    finally:
        settings.BETTERSTACK_BADGE_URL = prev
