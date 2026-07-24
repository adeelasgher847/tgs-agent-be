"""v2 enhanced health endpoint — public, no auth, with bounded service probes."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app


def test_v2_health_returns_200_without_auth():
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/api/v2/health")

    assert resp.status_code == 200


def test_v2_health_response_schema_all_ok():
    with patch(
        "app.api.v2.routers.health._probe_database", AsyncMock(return_value=True)
    ), patch(
        "app.api.v2.routers.health._probe_redis", AsyncMock(return_value=True)
    ), patch(
        "app.api.v2.routers.health._probe_voice_pipeline", AsyncMock(return_value=True)
    ):
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/api/v2/health")

    body = resp.json()
    assert body["status"] == "ok"
    assert body["services"] == {
        "api": True,
        "voice_pipeline": True,
        "database": True,
        "redis": True,
    }
    assert "timestamp" in body


def test_v2_health_degraded_when_redis_fails():
    with patch(
        "app.api.v2.routers.health._probe_database", AsyncMock(return_value=True)
    ), patch(
        "app.api.v2.routers.health._probe_redis", AsyncMock(return_value=False)
    ), patch(
        "app.api.v2.routers.health._probe_voice_pipeline", AsyncMock(return_value=True)
    ):
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/api/v2/health")

    body = resp.json()
    assert body["status"] == "degraded"
    assert body["services"]["redis"] is False
    assert body["services"]["database"] is True


def test_v2_health_degraded_when_voice_pipeline_times_out():
    async def _slow_health_check():
        await asyncio.sleep(2)
        return "ok"

    with patch(
        "app.api.v2.routers.health._probe_database", AsyncMock(return_value=True)
    ), patch(
        "app.api.v2.routers.health._probe_redis", AsyncMock(return_value=True)
    ), patch(
        "app.services.livekit_service.livekit_service.health_check",
        side_effect=_slow_health_check,
    ):
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/api/v2/health")

    body = resp.json()
    assert body["status"] == "degraded"
    assert body["services"]["voice_pipeline"] is False


def test_v2_health_down_when_database_fails():
    with patch(
        "app.api.v2.routers.health._probe_database", AsyncMock(return_value=False)
    ), patch(
        "app.api.v2.routers.health._probe_redis", AsyncMock(return_value=True)
    ), patch(
        "app.api.v2.routers.health._probe_voice_pipeline", AsyncMock(return_value=True)
    ):
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/api/v2/health")

    body = resp.json()
    assert body["status"] == "down"
    assert body["services"]["database"] is False
