"""
Shared Redis clients for the application (async + sync).

Both return None when REDIS_URL is unset or the connection fails, allowing
callers to skip caching gracefully.
"""
from __future__ import annotations

from typing import Optional

_redis = None
_redis_sync = None


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


def get_redis_sync():
    """Return the shared sync redis client, creating it on first call.

    Used by RBAC dependency functions, which are plain ``def`` (not
    ``async def``) to match the existing sync-Session dependency chain in
    app/api/deps.py — mixing in the asyncio client there would require
    awaiting inside a sync callable.
    """
    global _redis_sync
    if _redis_sync is not None:
        return _redis_sync

    try:
        from app.core.config import settings
        import redis

        if not settings.REDIS_URL:
            return None
        _redis_sync = redis.Redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
    except Exception:
        return None

    return _redis_sync
