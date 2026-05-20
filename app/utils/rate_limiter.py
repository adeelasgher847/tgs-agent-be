"""
Rate limiter utilities.

The global sliding-window rate limiter runs as middleware
(app/middleware/rate_limit_middleware.py). This module provides the
init/close lifecycle hooks called from app lifespan and graceful_shutdown.

Per-route fastapi-limiter decorators (login_rate_limit, webhook_rate_limit,
api_rate_limit) are removed — the global middleware replaces them for /api/v1/*
routes. Login / webhook paths are excluded from the global limiter via
_SKIP_PREFIXES in rate_limit_middleware.py.
"""
from __future__ import annotations

import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


async def init_rate_limiter() -> None:
    """
    Warm the global rate-limit Redis connection.

    The connection is created lazily on first use inside RateLimitMiddleware,
    so this is a no-op placeholder for lifespan hooks.
    """
    if settings.RATE_LIMIT_ENABLED:
        logger.info("Global sliding-window rate limiter ready (Redis lazy-connect)")
    else:
        logger.info("Rate limiting disabled (RATE_LIMIT_ENABLED=False)")


async def close_rate_limiter() -> None:
    """Release the Redis connection held by RateLimitMiddleware."""
    try:
        from app.middleware.rate_limit_middleware import close_rate_limit_middleware

        await close_rate_limit_middleware()
        logger.info("Rate limit middleware Redis connection closed")
    except Exception as exc:
        logger.warning("Rate limit cleanup failed: %s", exc)
