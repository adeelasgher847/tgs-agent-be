"""
Auth middleware latency tests — ticket SLA: p95 < 5 ms.

Measures the middleware hot path (not full route handlers).

  1. Always-on (CI): in-memory Redis + warm cache.
  2. Optional (``RUN_AUTH_PERF_TESTS=1`` + reachable Redis): real Redis cache-hit SLA.

Run optional tier:
  RUN_AUTH_PERF_TESTS=1 pytest tests/middleware/auth/test_api_key_middleware_perf.py -v
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Awaitable, Callable, List
from unittest.mock import MagicMock, patch

import pytest
from starlette.requests import Request

from app.core.security import create_user_token
from app.core.workspace import Workspace
from app.middleware import api_key_middleware as mw

P95_LIMIT_MS = 5.0
WARMUP_ITERATIONS = 50
SAMPLE_ITERATIONS = 300


def _p95_ms(samples: List[float]) -> float:
    ordered = sorted(samples)
    idx = max(0, int(len(ordered) * 0.95) - 1)
    return ordered[idx]


def _run_async(coro):
    import asyncio

    return asyncio.run(coro)


async def _collect_samples(
    coro_factory: Callable[[], Awaitable[None]],
    *,
    warmup: int = WARMUP_ITERATIONS,
    samples: int = SAMPLE_ITERATIONS,
) -> List[float]:
    for _ in range(warmup):
        await coro_factory()
    timings: List[float] = []
    for _ in range(samples):
        start = time.perf_counter()
        await coro_factory()
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def _http_request(headers: dict[str, str]) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/protected",
        "headers": raw,
    }
    return Request(scope)


def _workspace_dict(tenant_id: uuid.UUID) -> dict:
    tid = str(tenant_id)
    return {
        "id": tid,
        "name": "Perf WS",
        "schema_name": "perf_schema",
        "status": "active",
        "credits": 1.0,
        "stripe_customer_id": None,
        "stripe_subscription_id": None,
    }


def _api_key_cache_payload(tenant_id: uuid.UUID, key_id: uuid.UUID) -> dict:
    return {
        "api_key_id": str(key_id),
        "tenant_id": str(tenant_id),
        "key_is_active": True,
        "workspace": _workspace_dict(tenant_id),
    }


@pytest.fixture
def in_memory_redis():
    store: dict[str, str] = {}

    async def fake_get(key: str):
        return store.get(key)

    async def fake_setex(key: str, ttl: int, value: str):
        store[key] = value

    async def fake_delete(key: str):
        store.pop(key, None)

    mock = MagicMock()
    mock.get = fake_get
    mock.setex = fake_setex
    mock.delete = fake_delete

    with patch.object(mw, "_get_redis", return_value=mock):
        yield store


class TestMiddlewarePerfInProcess:
    """Hermetic perf tests — no external Redis/Postgres required."""

    def test_p95_api_key_auth_cache_hit_under_5ms(self, in_memory_redis):
        tenant_id = uuid.uuid4()
        key_id = uuid.uuid4()
        raw_key = "perf-test-api-key"
        key_hash = mw._sha256(raw_key)

        async def setup_and_run():
            await mw._apikey_cache_set(
                key_hash,
                tenant_id,
                _api_key_cache_payload(tenant_id, key_id),
            )

            async def run_once() -> None:
                request = _http_request(
                    {
                        "x-api-key": raw_key,
                        "x-workspace-id": str(tenant_id),
                    }
                )
                assert await mw._try_api_key_auth(request) is True

            return await _collect_samples(run_once)

        timings = _run_async(setup_and_run())
        p95 = _p95_ms(timings)
        p50 = sorted(timings)[len(timings) // 2]
        assert p95 < P95_LIMIT_MS, (
            f"API key auth cache-hit p95={p95:.3f}ms > {P95_LIMIT_MS}ms (p50={p50:.3f}ms)"
        )

    def test_p95_resolve_api_key_cache_hit_under_5ms(self, in_memory_redis):
        tenant_id = uuid.uuid4()
        key_id = uuid.uuid4()
        key_hash = mw._sha256("resolve-perf-key")

        async def setup_and_run():
            await mw._apikey_cache_set(
                key_hash,
                tenant_id,
                _api_key_cache_payload(tenant_id, key_id),
            )

            async def run_once() -> None:
                result = await mw._resolve_api_key(key_hash, tenant_id)
                assert result is not None

            return await _collect_samples(run_once)

        timings = _run_async(setup_and_run())
        p95 = _p95_ms(timings)
        assert p95 < P95_LIMIT_MS, f"_resolve_api_key cache-hit p95={p95:.3f}ms"

    def test_p95_jwt_auth_cached_workspace_under_5ms(self):
        tenant_id = uuid.uuid4()
        user_id = uuid.uuid4()
        workspace = Workspace.from_mapping(_workspace_dict(tenant_id))
        token = create_user_token(
            user_id=user_id,
            email="perf@test.com",
            tenant_id=tenant_id,
            role="admin",
        )

        async def fake_load(workspace_id: uuid.UUID):
            return workspace

        async def run_benchmark():
            with patch.object(mw, "_load_workspace", side_effect=fake_load):

                async def run_once() -> None:
                    request = _http_request({"authorization": f"Bearer {token}"})
                    assert await mw._try_jwt_auth(request) is True

                return await _collect_samples(run_once)

        timings = _run_async(run_benchmark())
        p95 = _p95_ms(timings)
        assert p95 < P95_LIMIT_MS, f"JWT auth (cached workspace) p95={p95:.3f}ms"


def _redis_available() -> bool:
    if os.getenv("RUN_AUTH_PERF_TESTS", "").lower() not in ("1", "true", "yes"):
        return False
    try:
        import redis  # noqa: WPS433

        client = redis.from_url(mw.settings.REDIS_URL, socket_connect_timeout=1)
        client.ping()
        client.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _redis_available(), reason="Set RUN_AUTH_PERF_TESTS=1 and Redis up")
class TestMiddlewarePerfRealRedis:
    """Optional: real Redis cache-hit p95 (staging/local)."""

    def test_p95_api_key_auth_real_redis_cache_hit_under_5ms(self):
        tenant_id = uuid.uuid4()
        key_id = uuid.uuid4()
        raw_key = f"perf-redis-{uuid.uuid4().hex}"
        key_hash = mw._sha256(raw_key)

        async def run_benchmark():
            await mw._apikey_cache_set(
                key_hash,
                tenant_id,
                _api_key_cache_payload(tenant_id, key_id),
            )

            async def run_once() -> None:
                request = _http_request(
                    {
                        "x-api-key": raw_key,
                        "x-workspace-id": str(tenant_id),
                    }
                )
                assert await mw._try_api_key_auth(request) is True

            return await _collect_samples(run_once, warmup=20, samples=150)

        timings = _run_async(run_benchmark())
        p95 = _p95_ms(timings)
        assert p95 < P95_LIMIT_MS, f"Real Redis cache-hit p95={p95:.3f}ms"

        _run_async(mw.invalidate_api_key_cache(raw_key, tenant_id))
