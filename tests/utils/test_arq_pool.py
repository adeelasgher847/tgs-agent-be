"""
Tests for app/utils/arq_pool.py — shared ARQ Redis pool lifecycle.

arq is not installed in the test environment (it's a runtime-only dep).
All tests inject a mock 'arq' module via sys.modules so no real Redis is needed.

Coverage:
  - init_arq_pool creates the pool and stores it in the singleton
  - get_arq_pool returns the same object across calls
  - close_arq_pool calls aclose() and resets the singleton to None
  - init failure is non-fatal (singleton stays None)
  - close is idempotent (safe to call when pool is None)
  - _enqueue_batch_job uses the shared pool (no new pool per request)
  - _enqueue_batch_job falls back to a per-call pool when singleton is None
  - multiple enqueue calls share the same pool instance
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_pool() -> MagicMock:
    pool = MagicMock()
    pool.enqueue_job = AsyncMock()
    pool.aclose = AsyncMock()
    return pool


@contextmanager
def _mock_arq(pool: MagicMock | None = None):
    """
    Inject a fake 'arq' module into sys.modules for the duration of the block.

    Necessary because arq is not installed in the test env; all imports are
    local-inside-function so patching sys.modules intercepts them correctly.
    """
    if pool is None:
        pool = _make_mock_pool()

    mock_arq = MagicMock()
    mock_arq.create_pool = AsyncMock(return_value=pool)
    mock_arq.connections = MagicMock()
    mock_arq.connections.RedisSettings.from_dsn = MagicMock(return_value=MagicMock())

    with patch.dict(sys.modules, {"arq": mock_arq, "arq.connections": mock_arq.connections}):
        yield mock_arq


@contextmanager
def _set_pool(value):
    """Temporarily set app.utils.arq_pool._pool and restore it afterwards."""
    import app.utils.arq_pool as _m
    original = _m._pool
    _m._pool = value
    try:
        yield
    finally:
        _m._pool = original


# ── Pool lifecycle ────────────────────────────────────────────────────────────

class TestArqPoolInit:
    def test_init_creates_pool_and_stores_singleton(self):
        """init_arq_pool must call arq.create_pool and store the result."""
        mock_pool = _make_mock_pool()

        async def _run():
            import app.utils.arq_pool as _m
            _m._pool = None
            with _mock_arq(mock_pool) as mock_arq:
                from app.utils.arq_pool import init_arq_pool
                await init_arq_pool()
                assert _m._pool is mock_pool
                mock_arq.create_pool.assert_awaited_once()

        asyncio.run(_run())

    def test_get_arq_pool_returns_stored_singleton(self):
        """get_arq_pool must return the value stored by init_arq_pool."""
        mock_pool = _make_mock_pool()

        async def _run():
            with _set_pool(mock_pool):
                from app.utils.arq_pool import get_arq_pool
                assert get_arq_pool() is mock_pool

        asyncio.run(_run())

    def test_get_arq_pool_returns_same_instance_on_repeated_calls(self):
        """Multiple get_arq_pool() calls must return the identical object."""
        mock_pool = _make_mock_pool()

        async def _run():
            with _set_pool(mock_pool):
                from app.utils.arq_pool import get_arq_pool
                assert get_arq_pool() is get_arq_pool()

        asyncio.run(_run())

    def test_init_failure_leaves_singleton_as_none(self):
        """A broken Redis URL must not raise — singleton stays None."""

        async def _run():
            import app.utils.arq_pool as _m
            _m._pool = None
            mock_arq = MagicMock()
            mock_arq.create_pool = AsyncMock(side_effect=ConnectionError("refused"))
            mock_arq.connections = MagicMock()
            mock_arq.connections.RedisSettings.from_dsn = MagicMock(return_value=MagicMock())
            with patch.dict(sys.modules, {"arq": mock_arq, "arq.connections": mock_arq.connections}):
                from app.utils.arq_pool import init_arq_pool
                await init_arq_pool()  # must not raise
            assert _m._pool is None

        asyncio.run(_run())


class TestArqPoolClose:
    def test_close_calls_aclose_and_resets_singleton(self):
        """close_arq_pool must call pool.aclose() and set _pool back to None."""
        mock_pool = _make_mock_pool()

        async def _run():
            import app.utils.arq_pool as _m
            _m._pool = mock_pool
            from app.utils.arq_pool import close_arq_pool
            await close_arq_pool()
            mock_pool.aclose.assert_awaited_once()
            assert _m._pool is None

        asyncio.run(_run())

    def test_close_is_idempotent_when_pool_is_none(self):
        """close_arq_pool with no pool must be a no-op and not raise."""

        async def _run():
            with _set_pool(None):
                from app.utils.arq_pool import close_arq_pool
                await close_arq_pool()  # must not raise

        asyncio.run(_run())

    def test_close_swallows_aclose_error(self):
        """If pool.aclose() raises, close_arq_pool must still reset singleton."""
        mock_pool = _make_mock_pool()
        mock_pool.aclose = AsyncMock(side_effect=RuntimeError("network gone"))

        async def _run():
            import app.utils.arq_pool as _m
            _m._pool = mock_pool
            from app.utils.arq_pool import close_arq_pool
            await close_arq_pool()  # must not raise
            assert _m._pool is None

        asyncio.run(_run())


# ── Enqueue uses shared pool ──────────────────────────────────────────────────

class TestEnqueueUsesSharedPool:
    def test_enqueue_uses_shared_pool_not_new_pool(self):
        """
        When the singleton is set, _enqueue_batch_job must use it directly
        without calling arq.create_pool.
        """
        mock_pool = _make_mock_pool()

        async def _run():
            with _set_pool(mock_pool):
                from app.api.v2.routers.batch_calls import _enqueue_batch_job
                with _mock_arq() as mock_arq:
                    await _enqueue_batch_job(str(uuid.uuid4()), None)
                    mock_arq.create_pool.assert_not_awaited()  # no new pool created
                mock_pool.enqueue_job.assert_awaited_once()
                mock_pool.aclose.assert_not_awaited()  # shared pool must not be closed

        asyncio.run(_run())

    def test_enqueue_falls_back_to_per_request_pool_when_singleton_is_none(self):
        """
        When the singleton is None, _enqueue_batch_job must create a temporary
        pool, use it, then close it.
        """
        fallback_pool = _make_mock_pool()

        async def _run():
            with _set_pool(None):
                from app.api.v2.routers.batch_calls import _enqueue_batch_job
                with _mock_arq(fallback_pool) as mock_arq:
                    await _enqueue_batch_job(str(uuid.uuid4()), None)
                    mock_arq.create_pool.assert_awaited_once()  # fallback pool created
                fallback_pool.enqueue_job.assert_awaited_once()
                fallback_pool.aclose.assert_awaited_once()  # per-request pool closed

        asyncio.run(_run())

    def test_multiple_requests_share_same_pool_instance(self):
        """
        Three sequential enqueue calls must all hit the same pool, never
        calling arq.create_pool.
        """
        mock_pool = _make_mock_pool()

        async def _run():
            with _set_pool(mock_pool):
                from app.api.v2.routers.batch_calls import _enqueue_batch_job
                with _mock_arq() as mock_arq:
                    for _ in range(3):
                        await _enqueue_batch_job(str(uuid.uuid4()), None)
                    mock_arq.create_pool.assert_not_awaited()
                assert mock_pool.enqueue_job.await_count == 3
                mock_pool.aclose.assert_not_awaited()

        asyncio.run(_run())

    def test_enqueue_with_future_scheduled_at_passes_defer_until(self):
        """A future scheduled_at must be forwarded as _defer_until to enqueue_job."""
        mock_pool = _make_mock_pool()
        future_dt = datetime(2099, 1, 1, tzinfo=timezone.utc)

        async def _run():
            with _set_pool(mock_pool):
                from app.api.v2.routers.batch_calls import _enqueue_batch_job
                await _enqueue_batch_job(str(uuid.uuid4()), future_dt)
                _, kwargs = mock_pool.enqueue_job.call_args
                assert "_defer_until" in kwargs
                assert kwargs["_defer_until"] == future_dt

        asyncio.run(_run())

    def test_enqueue_error_is_swallowed(self):
        """If enqueue_job raises, _enqueue_batch_job must not propagate the error."""
        mock_pool = _make_mock_pool()
        mock_pool.enqueue_job = AsyncMock(side_effect=ConnectionError("Redis gone"))

        async def _run():
            with _set_pool(mock_pool):
                from app.api.v2.routers.batch_calls import _enqueue_batch_job
                await _enqueue_batch_job(str(uuid.uuid4()), None)  # must not raise

        asyncio.run(_run())

    def test_startup_initializes_pool_and_shutdown_closes_it(self):
        """
        Full startup → use → shutdown sequence.

        init_arq_pool stores a pool; enqueue uses it; close_arq_pool releases it.
        """
        mock_pool = _make_mock_pool()

        async def _run():
            import app.utils.arq_pool as _m
            original = _m._pool
            try:
                _m._pool = None

                # 1. Startup
                with _mock_arq(mock_pool):
                    from app.utils.arq_pool import init_arq_pool
                    await init_arq_pool()
                assert _m._pool is mock_pool

                # 2. Request — should use shared pool, not create new one
                from app.api.v2.routers.batch_calls import _enqueue_batch_job
                with _mock_arq() as mock_arq_fallback:
                    await _enqueue_batch_job(str(uuid.uuid4()), None)
                    mock_arq_fallback.create_pool.assert_not_awaited()
                mock_pool.enqueue_job.assert_awaited_once()

                # 3. Shutdown
                from app.utils.arq_pool import close_arq_pool
                await close_arq_pool()
                mock_pool.aclose.assert_awaited_once()
                assert _m._pool is None
            finally:
                _m._pool = original

        asyncio.run(_run())
