"""Tests for per-IP login rate limiting (enforce_login_rate_limit)."""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.exception_handlers import register_exception_handlers
from app.middleware.request_id_middleware import RequestIdMiddleware
from app.utils.rate_limiter import enforce_login_rate_limit


@pytest.fixture(autouse=True)
def enable_rate_limiting():
    prev = settings.RATE_LIMIT_ENABLED
    settings.RATE_LIMIT_ENABLED = True
    try:
        yield
    finally:
        settings.RATE_LIMIT_ENABLED = prev


def _make_app() -> FastAPI:
    mini = FastAPI()
    register_exception_handlers(mini)
    mini.add_middleware(RequestIdMiddleware)

    @mini.post("/api/v1/users/login")
    async def login(_: None = Depends(enforce_login_rate_limit)):
        return {"ok": True}

    return mini


def _fake_redis_pipeline(count: int):
    class _Pipe:
        def zremrangebyscore(self, *a, **kw):
            return self

        def zadd(self, *a, **kw):
            return self

        def zcard(self, *a, **kw):
            return self

        def zrange(self, *a, **kw):
            return self

        def expire(self, *a, **kw):
            return self

        async def execute(self):
            oldest_score = time.time() - 50
            return [0, 1, count, [(f"entry:{time.time()}", oldest_score)], True]

    return _Pipe


class TestLoginRateLimit:
    def test_within_limit_passes(self):
        app = _make_app()
        mock_r = MagicMock()
        mock_r.pipeline.return_value = _fake_redis_pipeline(1)()

        with patch("app.middleware.rate_limit_middleware._get_redis", return_value=mock_r):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/api/v1/users/login")

        assert resp.status_code == 200

    def test_exceeding_limit_returns_429_with_envelope(self):
        app = _make_app()
        mock_r = MagicMock()
        mock_r.pipeline.return_value = _fake_redis_pipeline(999)()
        mock_r.zremrangebyscore = AsyncMock()

        with patch("app.middleware.rate_limit_middleware._get_redis", return_value=mock_r):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/api/v1/users/login")

        assert resp.status_code == 429
        body = resp.json()
        assert body["error"]["code"] == "rate_limit_exceeded"
        assert "retryAfter" in body["error"]

    def test_disabled_bypasses_check(self):
        app = _make_app()
        mock_r = MagicMock()
        from app.core.config import settings

        with patch.object(settings, "RATE_LIMIT_ENABLED", False):
            with patch("app.middleware.rate_limit_middleware._get_redis", return_value=mock_r):
                client = TestClient(app, raise_server_exceptions=False)
                resp = client.post("/api/v1/users/login")

        assert resp.status_code == 200
        mock_r.pipeline.assert_not_called()
