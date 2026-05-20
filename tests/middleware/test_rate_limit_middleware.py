"""
Unit tests for RateLimitMiddleware — sliding window, 429 envelope, skip paths.

Uses fakeredis so no real Redis is needed.
"""
from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.core.config import settings
from app.middleware.rate_limit_middleware import RateLimitMiddleware, _sha256
from app.middleware.request_id_middleware import RequestIdMiddleware


def _make_app(limit: int = 5, window: int = 60) -> FastAPI:
    mini = FastAPI()
    mini.add_middleware(RateLimitMiddleware)
    mini.add_middleware(RequestIdMiddleware)

    @mini.get("/api/v1/protected")
    def protected(request: Request):
        return {"ok": True}

    @mini.get("/health")
    def health():
        return {"ok": True}

    @mini.post("/api/v1/users/login")
    def login():
        return {"ok": True}

    return mini


def _fake_redis_pipeline(counts: list[int]):
    """
    Return a factory that hands back a pipeline mock.
    `counts` is a list of zcard results to return on successive pipeline.execute() calls.
    """
    call_idx = {"n": 0}

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
            i = call_idx["n"]
            call_idx["n"] += 1
            count = counts[i] if i < len(counts) else counts[-1]
            # [zremrangebyscore result, zadd result, zcard result, zrange result, expire result]
            oldest_score = time.time() - 50  # 10 s until slot frees
            return [0, 1, count, [(f"entry:{time.time()}", oldest_score)], True]

    return _Pipe


# ---------------------------------------------------------------------------
# Skip paths
# ---------------------------------------------------------------------------

class TestSkipPaths:
    def test_health_not_rate_limited(self):
        app = _make_app()
        with patch("app.middleware.rate_limit_middleware._get_redis", return_value=None):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/health")
        assert resp.status_code == 200

    def test_login_not_rate_limited(self):
        app = _make_app()
        with patch("app.middleware.rate_limit_middleware._get_redis", return_value=None):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/api/v1/users/login")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Allowed requests pass through
# ---------------------------------------------------------------------------

class TestAllowed:
    def test_request_within_limit_passes(self):
        app = _make_app()
        pipe_cls = _fake_redis_pipeline([1])  # count=1, limit=60 default

        mock_r = MagicMock()
        mock_r.pipeline.return_value = pipe_cls()
        mock_r.zremrangebyscore = AsyncMock()

        with patch("app.middleware.rate_limit_middleware._get_redis", return_value=mock_r):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/protected")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Rate limit trigger → 429
# ---------------------------------------------------------------------------

class TestRateLimitExceeded:
    def test_exceeding_limit_returns_429(self):
        """Simulate a count > limit from Redis → expect 429."""
        app = _make_app()
        pipe_cls = _fake_redis_pipeline([61])  # count > 60 default limit

        mock_r = MagicMock()
        mock_r.pipeline.return_value = pipe_cls()
        mock_r.zremrangebyscore = AsyncMock()

        with patch("app.middleware.rate_limit_middleware._get_redis", return_value=mock_r):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/protected")

        assert resp.status_code == 429

    def test_429_envelope_shape(self):
        app = _make_app()
        pipe_cls = _fake_redis_pipeline([999])

        mock_r = MagicMock()
        mock_r.pipeline.return_value = pipe_cls()
        mock_r.zremrangebyscore = AsyncMock()

        with patch("app.middleware.rate_limit_middleware._get_redis", return_value=mock_r):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/protected")

        assert resp.status_code == 429
        body = resp.json()
        assert body["error"]["code"] == "rate_limit_exceeded"
        assert "retry" in body["error"]["message"].lower()
        assert "retryAfter" in body["error"]
        # retryAfter must be a non-empty ISO string
        retry = body["error"]["retryAfter"]
        assert isinstance(retry, str) and retry.endswith("Z")
        assert set(body["error"].keys()) == {"code", "message", "retryAfter"}

    def test_429_has_request_id_header(self):
        app = _make_app()
        pipe_cls = _fake_redis_pipeline([999])

        mock_r = MagicMock()
        mock_r.pipeline.return_value = pipe_cls()
        mock_r.zremrangebyscore = AsyncMock()

        with patch("app.middleware.rate_limit_middleware._get_redis", return_value=mock_r):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/protected")

        assert "x-request-id" in resp.headers

    def test_no_redis_passes_through(self):
        """When Redis is unavailable, rate limiter must fail open (allow)."""
        app = _make_app()
        with patch("app.middleware.rate_limit_middleware._get_redis", return_value=None):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/protected")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# SHA-256 helper (reused from api_key_middleware)
# ---------------------------------------------------------------------------

class TestRateLimitDisabled:
    def test_disabled_bypasses_redis(self):
        app = _make_app()
        mock_r = MagicMock()
        with patch.object(settings, "RATE_LIMIT_ENABLED", False):
            with patch("app.middleware.rate_limit_middleware._get_redis", return_value=mock_r):
                client = TestClient(app, raise_server_exceptions=False)
                resp = client.get("/api/v1/protected")
        assert resp.status_code == 200
        mock_r.pipeline.assert_not_called()


class TestSha256:
    def test_sha256_produces_64_hex_chars(self):
        assert len(_sha256("key")) == 64

    def test_sha256_deterministic(self):
        assert _sha256("same") == _sha256("same")

    def test_sha256_different_inputs(self):
        assert _sha256("a") != _sha256("b")
