"""
Shared ARQ Redis pool.

Initialized once at application startup (init_arq_pool) and closed at shutdown
(close_arq_pool).  All request handlers call get_arq_pool() to obtain the
singleton — no connection overhead per request.

Follows the same singleton pattern as app/middleware/rate_limit_middleware.py.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from app.core.logger import logger

if TYPE_CHECKING:
    import arq  # noqa: F401 — type-only import; arq may not be installed in all envs

_pool: Optional[object] = None  # arq.ArqRedis at runtime


async def init_arq_pool() -> None:
    """
    Create the shared ARQ Redis pool and store it in the module singleton.

    Called once from the FastAPI lifespan.  A failure here is non-fatal:
    batch job enqueueing falls back to a per-request pool so the batch upload
    endpoint still returns 201 — the ARQ worker self-heals on its next 60-s poll.
    """
    global _pool
    try:
        import arq as _arq
        from app.core.config import settings

        redis_settings = _arq.connections.RedisSettings.from_dsn(settings.REDIS_URL)
        _pool = await _arq.create_pool(redis_settings)
        logger.info("ARQ Redis pool initialized successfully")
    except Exception as exc:
        logger.warning(
            "ARQ pool initialization failed: %s — batch jobs will fall back to per-request pool",
            exc,
        )


async def close_arq_pool() -> None:
    """
    Close the shared ARQ Redis pool gracefully.

    Called from graceful_shutdown().  Errors are swallowed so other teardown
    steps are not interrupted.
    """
    global _pool
    if _pool is not None:
        try:
            await _pool.aclose()  # type: ignore[union-attr]
            logger.info("ARQ Redis pool closed")
        except Exception as exc:
            logger.warning("ARQ pool close failed: %s", exc)
        finally:
            _pool = None


def get_arq_pool() -> Optional[object]:
    """Return the shared pool, or None if it was not (yet) initialized."""
    return _pool
