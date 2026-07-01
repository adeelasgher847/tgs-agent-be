"""
Rate limiter integration tests against real PostgreSQL + real Redis.

Requires TEST_DATABASE_URL (for pg_client auth) AND a reachable Redis instance
at settings.REDIS_URL. Tests are skipped individually if Redis is not reachable.

Exercises the full sliding-window rate-limit path end-to-end:
  ApiKeyMiddleware (PG auth) → RateLimitMiddleware (Redis) → route handler

Coverage:
  1.  test_requests_within_limit_are_allowed
  2.  test_request_exceeding_limit_returns_429
  3.  test_429_response_body_has_correct_structure
  4.  test_429_response_has_x_request_id_header
  5.  test_rate_limit_is_per_identity_not_global
  6.  test_rate_limit_disabled_flag_bypasses_redis
"""
from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from app.core.config import settings
from app.models.tenant import Tenant
from app.services.api_key_service import create_api_key
from app.middleware.rate_limit_middleware import _sha256
from tests.conftest import _INTEGRATION_SKIP

pytestmark = [_INTEGRATION_SKIP, pytest.mark.integration]

# ---------------------------------------------------------------------------
# Redis availability skip — checked once per session via a fixture.
# The sync redis client does a quick ping; if it fails the fixture skips the test.
# ---------------------------------------------------------------------------

_REDIS_SKIP_REASON = "Redis not reachable — skipping rate-limiter integration tests"


@pytest.fixture(scope="session")
def redis_sync_client():
    """Session-scoped sync Redis client. Skips if Redis is not reachable."""
    try:
        import redis as _sync_redis
    except ImportError:
        pytest.skip("redis package not installed")

    url = settings.REDIS_URL
    try:
        r = _sync_redis.from_url(url, socket_connect_timeout=1, socket_timeout=1)
        r.ping()
    except Exception as exc:
        pytest.skip(f"{_REDIS_SKIP_REASON}: {exc}")

    yield r
    r.close()


# ---------------------------------------------------------------------------
# Auth fixtures — same pattern as test_workspace_postgres.py
# ---------------------------------------------------------------------------


@pytest.fixture()
def auth_workspace(pg_session):
    """Tenant + real API key committed in the Postgres test schema."""
    tenant = Tenant(
        name=f"RLPg-{uuid.uuid4().hex[:8]}",
        schema_name=f"rl_pg_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    pg_session.add(tenant)
    pg_session.commit()
    pg_session.refresh(tenant)

    _record, raw_key = create_api_key(
        pg_session, tenant_id=tenant.id, name="rate-limit-integration-test"
    )
    return tenant, raw_key


def _headers(raw_key: str, tenant_id: uuid.UUID) -> dict[str, str]:
    return {"x-api-key": raw_key, "x-workspace-id": str(tenant_id)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flush_rate_key(redis_client, raw_key: str) -> None:
    """Delete the sliding-window Redis key for this API key identity."""
    redis_client.delete(f"rate_limit:{_sha256(raw_key)}")


def _hit_endpoint(pg_client, raw_key: str, tenant_id: uuid.UUID):
    """Single GET /api/v1/workspace/{id} — simple authenticated endpoint."""
    return pg_client.get(
        f"/api/v1/workspace/{tenant_id}",
        headers=_headers(raw_key, tenant_id),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRateLimiterPostgres:

    def test_requests_within_limit_are_allowed(
        self, pg_client, redis_sync_client, auth_workspace
    ):
        tenant, raw_key = auth_workspace
        _flush_rate_key(redis_sync_client, raw_key)

        with (
            patch.object(settings, "RATE_LIMIT_ENABLED", True),
            patch.object(settings, "API_RATE_LIMIT", 5),
            patch("app.middleware.rate_limit_middleware._redis", None),
        ):
            for _ in range(5):
                resp = _hit_endpoint(pg_client, raw_key, tenant.id)
                assert resp.status_code == 200, resp.text

        _flush_rate_key(redis_sync_client, raw_key)

    def test_request_exceeding_limit_returns_429(
        self, pg_client, redis_sync_client, auth_workspace
    ):
        tenant, raw_key = auth_workspace
        _flush_rate_key(redis_sync_client, raw_key)

        with (
            patch.object(settings, "RATE_LIMIT_ENABLED", True),
            patch.object(settings, "API_RATE_LIMIT", 3),
            patch("app.middleware.rate_limit_middleware._redis", None),
        ):
            responses = [
                _hit_endpoint(pg_client, raw_key, tenant.id)
                for _ in range(4)
            ]

        statuses = [r.status_code for r in responses]
        # First 3 must pass; the 4th must be rate-limited
        assert statuses[:3] == [200, 200, 200]
        assert statuses[3] == 429

        _flush_rate_key(redis_sync_client, raw_key)

    def test_429_response_body_has_correct_structure(
        self, pg_client, redis_sync_client, auth_workspace
    ):
        tenant, raw_key = auth_workspace
        _flush_rate_key(redis_sync_client, raw_key)

        with (
            patch.object(settings, "RATE_LIMIT_ENABLED", True),
            patch.object(settings, "API_RATE_LIMIT", 2),
            patch("app.middleware.rate_limit_middleware._redis", None),
        ):
            for _ in range(2):
                _hit_endpoint(pg_client, raw_key, tenant.id)
            resp = _hit_endpoint(pg_client, raw_key, tenant.id)

        assert resp.status_code == 429
        body = resp.json()
        assert "error" in body
        err = body["error"]
        assert err["code"] == "rate_limit_exceeded"
        assert "retry" in err["message"].lower()
        assert "retryAfter" in err
        assert isinstance(err["retryAfter"], str) and err["retryAfter"].endswith("Z")
        assert set(err.keys()) == {"code", "message", "retryAfter"}

        _flush_rate_key(redis_sync_client, raw_key)

    def test_429_response_has_x_request_id_header(
        self, pg_client, redis_sync_client, auth_workspace
    ):
        tenant, raw_key = auth_workspace
        _flush_rate_key(redis_sync_client, raw_key)

        with (
            patch.object(settings, "RATE_LIMIT_ENABLED", True),
            patch.object(settings, "API_RATE_LIMIT", 2),
            patch("app.middleware.rate_limit_middleware._redis", None),
        ):
            for _ in range(2):
                _hit_endpoint(pg_client, raw_key, tenant.id)
            resp = _hit_endpoint(pg_client, raw_key, tenant.id)

        assert resp.status_code == 429
        assert "x-request-id" in resp.headers

        _flush_rate_key(redis_sync_client, raw_key)

    def test_rate_limit_is_per_identity_not_global(
        self, pg_client, redis_sync_client, pg_session, auth_workspace
    ):
        """A second tenant with a different API key has its own independent window."""
        tenant_a, key_a = auth_workspace

        tenant_b = Tenant(
            name=f"RLPg-B-{uuid.uuid4().hex[:8]}",
            schema_name=f"rl_pg_b_{uuid.uuid4().hex[:8]}",
            status="active",
        )
        pg_session.add(tenant_b)
        pg_session.commit()
        pg_session.refresh(tenant_b)
        _, key_b = create_api_key(pg_session, tenant_id=tenant_b.id, name="rl-b")

        _flush_rate_key(redis_sync_client, key_a)
        _flush_rate_key(redis_sync_client, key_b)

        with (
            patch.object(settings, "RATE_LIMIT_ENABLED", True),
            patch.object(settings, "API_RATE_LIMIT", 2),
            patch("app.middleware.rate_limit_middleware._redis", None),
        ):
            # Exhaust identity A's quota
            for _ in range(3):
                _hit_endpoint(pg_client, key_a, tenant_a.id)

            # Identity B should still be allowed
            resp_b = _hit_endpoint(pg_client, key_b, tenant_b.id)

        assert resp_b.status_code == 200, (
            "Identity B should not be blocked by identity A exhausting its quota"
        )

        _flush_rate_key(redis_sync_client, key_a)
        _flush_rate_key(redis_sync_client, key_b)

    def test_rate_limit_disabled_flag_bypasses_redis(
        self, pg_client, redis_sync_client, auth_workspace
    ):
        """When RATE_LIMIT_ENABLED=False, the middleware must not touch Redis
        even if a real Redis is available — the flag is the authoritative gate."""
        tenant, raw_key = auth_workspace
        _flush_rate_key(redis_sync_client, raw_key)

        with patch.object(settings, "RATE_LIMIT_ENABLED", False):
            for _ in range(20):
                resp = _hit_endpoint(pg_client, raw_key, tenant.id)
                assert resp.status_code == 200, "All requests should pass when rate limit disabled"

        _flush_rate_key(redis_sync_client, raw_key)
