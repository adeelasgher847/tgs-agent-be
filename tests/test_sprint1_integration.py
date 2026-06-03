"""Sprint 1 integration tests — run against the full app.main middleware stack.

Coverage:
  - GET /health on full app
  - 404 handler produces error envelope
  - Error middleware propagates x-request-id
  - Rate limiter returns 429 after limit+1 HTTP requests (mocked Redis counts)

Uses SQLite from conftest.py — no Postgres required.
"""
from __future__ import annotations

import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


# ---------------------------------------------------------------------------
# Shared fixture: full-app TestClient
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def full_client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# Health endpoint on full app.main
# ---------------------------------------------------------------------------

class TestHealthFullApp:
    def test_health_returns_200(self, full_client):
        resp = full_client.get("/health")
        assert resp.status_code == 200

    def test_health_body_shape(self, full_client):
        body = full_client.get("/health").json()
        assert body["status"] == "ok"
        assert "version" in body
        assert "timestamp" in body

    def test_health_not_wrapped_in_success_response(self, full_client):
        body = full_client.get("/health").json()
        assert "data" not in body
        assert "message" not in body

    def test_health_timestamp_is_utc_iso(self, full_client):
        from datetime import datetime
        ts = full_client.get("/health").json()["timestamp"]
        dt = datetime.fromisoformat(ts)
        assert dt.tzinfo is not None


# ---------------------------------------------------------------------------
# 404 handler on full app.main
# ---------------------------------------------------------------------------

class TestNotFoundFullApp:
    def test_unknown_route_returns_404(self, full_client):
        resp = full_client.get("/this-does-not-exist")
        assert resp.status_code == 404

    def test_404_error_envelope_shape(self, full_client):
        body = full_client.get("/nonexistent-route").json()
        assert "error" in body
        err = body["error"]
        assert err["code"] == "not_found"
        assert "message" in err
        assert "requestId" in err

    def test_404_has_request_id_header(self, full_client):
        resp = full_client.get("/nonexistent-route")
        assert "x-request-id" in resp.headers

    def test_request_id_consistent_in_header_and_body(self, full_client):
        resp = full_client.get("/nonexistent-route")
        rid_header = resp.headers.get("x-request-id", "")
        rid_body = resp.json()["error"]["requestId"]
        assert rid_header == rid_body
        assert rid_header != ""

    def test_custom_request_id_propagated(self, full_client):
        custom_id = f"test-{uuid.uuid4().hex[:8]}"
        resp = full_client.get("/nonexistent-route", headers={"x-request-id": custom_id})
        assert resp.json()["error"]["requestId"] == custom_id


# ---------------------------------------------------------------------------
# Error middleware envelope on full app.main
# ---------------------------------------------------------------------------

class TestErrorMiddlewareFullApp:
    def test_missing_auth_returns_401_envelope(self, full_client):
        resp = full_client.get(f"/api/v1/workspace/{uuid.uuid4()}")
        assert resp.status_code == 401
        body = resp.json()
        assert "error" in body
        assert body["error"]["code"] == "unauthorized"
        assert "requestId" in body["error"]

    def test_401_has_request_id_header(self, full_client):
        resp = full_client.get(f"/api/v1/workspace/{uuid.uuid4()}")
        assert "x-request-id" in resp.headers


# ---------------------------------------------------------------------------
# Rate limiter — limit+1 real HTTP requests (mocked Redis zcard sequence)
# ---------------------------------------------------------------------------

def _mock_redis_with_counts(counts: list[int]) -> MagicMock:
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
            return [0, 1, count, [(f"entry:{time.time()}", time.time() - 50)], True]

    mock_r = MagicMock()
    mock_r.pipeline.return_value = _Pipe()
    mock_r.zremrangebyscore = AsyncMock()
    return mock_r


@pytest.fixture()
def rate_limited_client(full_client):
    """Patch Redis so the sliding-window count reaches the limit on the last request."""
    with patch("app.middleware.rate_limit_middleware._get_redis") as get_rl_redis:
        with patch("app.middleware.api_key_middleware._get_redis", return_value=None):
            yield full_client, get_rl_redis


class TestRateLimitFullApp:
    """Make limit+1 authenticated requests through the real middleware stack."""

    def _make_authed_headers(self):
        import uuid as _uuid

        # Mocked auth only — no SQLite writes (avoids polluting workspace tests).
        tid = str(_uuid.uuid4())
        payload = {
            "api_key_id": str(_uuid.uuid4()),
            "tenant_id": tid,
            "key_is_active": True,
            "workspace": {
                "id": tid,
                "name": "RLTest",
                "schema_name": "rl_schema",
                "status": "active",
                "credits": 0.0,
                "stripe_customer_id": None,
                "stripe_subscription_id": None,
            },
        }
        return {"x-api-key": "sk_ratelimit_test_key", "x-workspace-id": tid}, payload

    def test_61st_request_returns_429(self, rate_limited_client):
        client, get_rl_redis = rate_limited_client
        headers, payload = self._make_authed_headers()

        async def _resolve(_key_hash, _workspace_id):
            return payload

        from app.core.config import settings
        limit = settings.API_RATE_LIMIT  # default 60
        counts = list(range(1, limit + 2))
        get_rl_redis.return_value = _mock_redis_with_counts(counts)

        with patch(
            "app.middleware.api_key_middleware._resolve_api_key",
            side_effect=_resolve,
        ):
            responses = []
            for _ in range(limit + 1):
                resp = client.get(
                    f"/api/v1/workspace/{payload['tenant_id']}",
                    headers=headers,
                )
                responses.append(resp.status_code)

        assert responses[-1] == 429, f"Expected 429 on request {limit+1}, got {responses[-1]}"

    def test_429_envelope_shape(self, rate_limited_client):
        client, get_rl_redis = rate_limited_client
        headers, payload = self._make_authed_headers()

        async def _resolve(_key_hash, _workspace_id):
            return payload

        from app.core.config import settings
        limit = settings.API_RATE_LIMIT
        get_rl_redis.return_value = _mock_redis_with_counts(list(range(1, limit + 2)))

        with patch(
            "app.middleware.api_key_middleware._resolve_api_key",
            side_effect=_resolve,
        ):
            last = None
            for _ in range(limit + 1):
                last = client.get(
                    f"/api/v1/workspace/{payload['tenant_id']}",
                    headers=headers,
                )

        assert last.status_code == 429
        body = last.json()
        assert body["error"]["code"] == "rate_limit_exceeded"
        assert "retryAfter" in body["error"]
        assert "x-request-id" in last.headers
