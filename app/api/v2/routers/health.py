"""
Public enhanced health endpoint — GET /api/v2/health.

No API key or workspace header required (see the skip list in
app/middleware/api_key_middleware.py). Each dependency probe below is bounded
to a 1-second timeout so a slow/unreachable downstream never makes this
endpoint itself time out — Betterstack (or any external monitor) polling this
route always gets a fast, well-formed response.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.async_session import get_db as get_async_db
from app.utils.redis_client import get_redis
from app.services.livekit_service import livekit_service

router = APIRouter(tags=["v2"])

PROBE_TIMEOUT_SECONDS = 1.0


async def _probe_database(db: AsyncSession) -> bool:
    try:
        await asyncio.wait_for(db.execute(text("SELECT 1")), timeout=PROBE_TIMEOUT_SECONDS)
        return True
    except asyncio.TimeoutError:
        # wait_for() cancels the in-flight query without the DBAPI driver's
        # knowledge; invalidate so the (possibly mid-read) connection is
        # discarded instead of returned to the pool for the next request.
        await db.invalidate()
        return False
    except Exception:
        return False


async def _probe_redis() -> bool:
    redis = get_redis()
    if redis is None:
        return False
    try:
        return bool(await asyncio.wait_for(redis.ping(), timeout=PROBE_TIMEOUT_SECONDS))
    except asyncio.TimeoutError:
        # Same rationale as the DB probe: a cancelled command can leave the
        # shared connection mid-read. get_redis() is also used by RBAC
        # caching and the live-call conversation orchestrator, so disconnect
        # the pool rather than let those callers read a desynced response.
        try:
            await redis.connection_pool.disconnect()
        except Exception:
            pass
        return False
    except Exception:
        return False


async def _probe_voice_pipeline() -> bool:
    try:
        result = await asyncio.wait_for(
            livekit_service.health_check(), timeout=PROBE_TIMEOUT_SECONDS
        )
        return result == "ok"
    except Exception:
        return False


@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_async_db)) -> dict:
    database_ok, redis_ok, voice_pipeline_ok = await asyncio.gather(
        _probe_database(db), _probe_redis(), _probe_voice_pipeline()
    )

    services = {
        "api": True,
        "voice_pipeline": voice_pipeline_ok,
        "database": database_ok,
        "redis": redis_ok,
    }

    if not database_ok:
        status = "down"
    elif all(services.values()):
        status = "ok"
    else:
        status = "degraded"

    return {
        "status": status,
        "services": services,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
