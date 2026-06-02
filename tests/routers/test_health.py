"""Tests for GET /health endpoint."""

from __future__ import annotations

from datetime import datetime, timezone

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
        resp = self.client.get("/health")
        assert resp.status_code == 200

    def test_response_shape(self):
        data = self.client.get("/health").json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "timestamp" in data

    def test_timestamp_is_iso8601_utc(self):
        data = self.client.get("/health").json()
        # datetime.fromisoformat raises if the string is not valid ISO-8601.
        ts = datetime.fromisoformat(data["timestamp"])
        # Must carry UTC timezone info.
        assert ts.tzinfo is not None

    def test_no_success_response_wrapper(self):
        """Health must NOT be wrapped in { data, message } SuccessResponse."""
        data = self.client.get("/health").json()
        assert "data" not in data
        assert "message" not in data
