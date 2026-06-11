"""
Shared async Redis client for the application.

Returns None when REDIS_URL is unset or the connection fails, allowing
callers to skip caching gracefully.
"""
from __future__ import annotations

from typing import Optional

_redis = None


def get_redis():
    """Return the shared aioredis client, creating it on first call."""
    global _redis
    if _redis is not None:
        return _redis

    try:
        from app.core.config import settings
        import redis.asyncio as aioredis

        if not settings.REDIS_URL:
            return None
        _redis = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
    except Exception:
        return None

    return _redis
