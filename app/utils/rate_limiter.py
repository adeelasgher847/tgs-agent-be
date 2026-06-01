"""
Rate limiter utilities.

Global sliding-window rate limiting runs in RateLimitMiddleware for /api/v1/* routes.
Auth-adjacent POST routes get a stricter per-IP limit in middleware; login/google
login also use enforce_login_rate_limit() on the route (same Redis sliding window).

Lifecycle hooks (init_rate_limiter / close_rate_limiter) are called from app lifespan
and graceful_shutdown.
"""
from __future__ import annotations

import logging

from fastapi import HTTPException, Request, status

from app.core.config import settings
from app.middleware.request_id_middleware import get_request_id

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


def _client_ip(request: Request) -> str:
    client = request.client
    return client.host if client else "unknown"


async def enforce_login_rate_limit(request: Request) -> None:
    """
    Stricter per-IP sliding-window limit for unauthenticated login endpoints.

    Uses Redis key ``rate_limit:login:{ip}`` with LOGIN_RATE_LIMIT / LOGIN_RATE_WINDOW.
    Middleware skips the global bucket for login paths to avoid double-counting.
    """
    if not settings.RATE_LIMIT_ENABLED:
        return

    from app.middleware.rate_limit_middleware import (
        _check_rate_limit,
        build_rate_limit_error,
    )

    key = f"login:{_client_ip(request)}"
    allowed, retry_after = await _check_rate_limit(
        key, settings.LOGIN_RATE_LIMIT, settings.LOGIN_RATE_WINDOW
    )
    if not allowed:
        error = build_rate_limit_error(retry_after)
        error["requestId"] = get_request_id(request)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=error,
        )
