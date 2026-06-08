"""Tests for GET /health endpoint."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.health import router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestHealthEndpoint:
    def setup_method(self):
        self.client = _client()

    def test_status_200(self):
        with patch(
            "app.routers.health.livekit_service.health_check",
            new=AsyncMock(return_value="ok"),
        ):
            resp = self.client.get("/health")
        assert resp.status_code == 200

    def test_response_shape(self):
        with patch(
            "app.routers.health.livekit_service.health_check",
            new=AsyncMock(return_value="ok"),
        ):
            data = self.client.get("/health").json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "timestamp" in data

    def test_timestamp_is_iso8601_utc(self):
        with patch(
            "app.routers.health.livekit_service.health_check",
            new=AsyncMock(return_value="ok"),
        ):
            data = self.client.get("/health").json()
        ts = datetime.fromisoformat(data["timestamp"])
        assert ts.tzinfo is not None

    def test_no_success_response_wrapper(self):
        """Health must NOT be wrapped in { data, message } SuccessResponse."""
        with patch(
            "app.routers.health.livekit_service.health_check",
            new=AsyncMock(return_value="ok"),
        ):
            data = self.client.get("/health").json()
        assert "data" not in data
        assert "message" not in data

    def test_livekit_key_present_in_response(self):
        """Response must include a 'livekit' key."""
        with patch(
            "app.routers.health.livekit_service.health_check",
            new=AsyncMock(return_value="ok"),
        ):
            data = self.client.get("/health").json()
        assert "livekit" in data

    def test_livekit_ok_when_livekit_enabled_false(self):
        """When LIVEKIT_ENABLED=False, health_check() returns 'ok' immediately."""
        with patch(
            "app.routers.health.livekit_service.health_check",
            new=AsyncMock(return_value="ok"),
        ):
            data = self.client.get("/health").json()
        assert data["livekit"] == "ok"

    def test_livekit_degraded_when_unreachable(self):
        """livekit='degraded' when LiveKit server is unreachable."""
        with patch(
            "app.routers.health.livekit_service.health_check",
            new=AsyncMock(return_value="degraded"),
        ):
            data = self.client.get("/health").json()
        assert data["livekit"] == "degraded"

    def test_health_never_returns_500_on_livekit_failure(self):
        """HTTP 500 must never be returned due to LiveKit failure."""
        with patch(
            "app.routers.health.livekit_service.health_check",
            new=AsyncMock(side_effect=Exception("LiveKit completely down")),
        ):
            resp = self.client.get("/health")
        # Even when health_check() raises unexpectedly, the endpoint must return 200
        # with livekit='degraded' — never propagate a 500.
        assert resp.status_code == 200
        assert resp.json()["livekit"] == "degraded"

    def test_status_ok_regardless_of_livekit(self):
        """top-level 'status' is always 'ok' regardless of LiveKit state."""
        for lk_status in ("ok", "degraded"):
            with patch(
                "app.routers.health.livekit_service.health_check",
                new=AsyncMock(return_value=lk_status),
            ):
                data = self.client.get("/health").json()
            assert data["status"] == "ok", (
                f"status must be 'ok' even when livekit='{lk_status}'"
            )
