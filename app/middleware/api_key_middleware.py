"""
API key authentication middleware.

Every request to /api/v1/* (except public paths) must supply:
  x-api-key: <raw key>
  x-workspace-id: <tenant UUID>

The raw key is SHA-256 hashed and looked up in the `apikey` table.
Valid lookups are cached in Redis for 60 s to keep p95 < 5 ms.
On success, request.state.workspace (Tenant) and request.state.api_key_id (UUID)
are set for downstream handlers.
X-Request-ID is injected on every response.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Optional

import redis.asyncio as aioredis
from fastapi import Request
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.logger import logger
from app.models.api_key import Apikey
from app.models.tenant import Tenant

# ---------------------------------------------------------------------------
# Redis client (shared, lazy-initialised)
# ---------------------------------------------------------------------------
_redis: Optional[aioredis.Redis] = None

CACHE_TTL = 60  # seconds


def _get_redis() -> Optional[aioredis.Redis]:
    global _redis
    if _redis is None:
        try:
            _redis = aioredis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("API key middleware: Redis unavailable (%s) — cache disabled", exc)
    return _redis


# ---------------------------------------------------------------------------
# Async DB engine (separate from the sync engine used by the rest of the app)
# ---------------------------------------------------------------------------
_async_engine = None
_AsyncSessionLocal: Optional[sessionmaker] = None


def _get_async_session() -> AsyncSession:
    global _async_engine, _AsyncSessionLocal
    if _async_engine is None:
        async_url = settings.DATABASE_URL.replace(
            "postgresql+psycopg2://", "postgresql+asyncpg://"
        ).replace("postgresql://", "postgresql+asyncpg://")
        from sqlalchemy.ext.asyncio import create_async_engine as _cae
        _async_engine = _cae(async_url, pool_pre_ping=True)
        _AsyncSessionLocal = sessionmaker(
            _async_engine, class_=AsyncSession, expire_on_commit=False
        )
    return _AsyncSessionLocal()


# ---------------------------------------------------------------------------
# Paths that bypass API-key auth entirely
# ---------------------------------------------------------------------------
_SKIP_EXACT = {"/", "/docs", "/redoc", "/openapi.json"}
_SKIP_PREFIXES = (
    "/api/v1/auth/",
    "/api/v1/billing/webhook",
    "/api/v1/voice/",          # Twilio webhooks — authenticated via Twilio signature
    "/api/v1/stream/",         # WebSocket media streams
    "/health",
    "/docs/",
    "/redoc/",
)


def _should_skip(path: str) -> bool:
    if path in _SKIP_EXACT:
        return True
    return any(path.startswith(prefix) for prefix in _SKIP_PREFIXES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def _err(status: int, detail: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"detail": detail},
        headers={"X-Request-ID": request_id},
    )


# ---------------------------------------------------------------------------
# Cache read / write
# ---------------------------------------------------------------------------

async def _cache_get(key_hash: str) -> Optional[dict]:
    r = _get_redis()
    if r is None:
        return None
    try:
        raw = await r.get(f"apikey:{key_hash}")
        return json.loads(raw) if raw else None
    except Exception as exc:
        logger.debug("Redis get failed: %s", exc)
        return None


async def _cache_set(key_hash: str, payload: dict) -> None:
    r = _get_redis()
    if r is None:
        return
    try:
        await r.setex(f"apikey:{key_hash}", CACHE_TTL, json.dumps(payload))
    except Exception as exc:
        logger.debug("Redis set failed: %s", exc)


async def _cache_delete(key_hash: str) -> None:
    r = _get_redis()
    if r is None:
        return
    try:
        await r.delete(f"apikey:{key_hash}")
    except Exception as exc:
        logger.debug("Redis delete failed: %s", exc)


# ---------------------------------------------------------------------------
# Core lookup: cache → DB
# ---------------------------------------------------------------------------

async def _resolve_api_key(key_hash: str, workspace_id: uuid.UUID) -> Optional[dict]:
    """
    Returns a dict with tenant/key info on success, None on any miss.
    Dict shape: {api_key_id, tenant_id, tenant_status, key_is_active}
    """
    cached = await _cache_get(key_hash)
    if cached is not None:
        return cached

    try:
        async with _get_async_session() as session:
            result = await session.execute(
                select(Apikey, Tenant)
                .join(Tenant, Apikey.tenant_id == Tenant.id)
                .where(Apikey.key_hash == key_hash)
            )
            row = result.first()
    except Exception as exc:
        logger.error("API key middleware DB lookup failed: %s", exc, exc_info=True)
        return None

    if row is None:
        return None

    api_key_obj, tenant_obj = row
    payload = {
        "api_key_id": str(api_key_obj.id),
        "tenant_id": str(tenant_obj.id),
        "tenant_status": tenant_obj.status,
        "key_is_active": api_key_obj.is_active,
    }
    await _cache_set(key_hash, payload)
    return payload


# ---------------------------------------------------------------------------
# ASGI middleware class
# ---------------------------------------------------------------------------

class ApiKeyMiddleware:
    """ASGI middleware that enforces x-api-key / x-workspace-id on API routes."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket" or scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        scope["_request_id"] = request_id

        if not request.url.path.startswith("/api/v1/") or _should_skip(request.url.path):
            await self._call_with_request_id(scope, receive, send, request_id)
            return

        raw_key = request.headers.get("x-api-key", "").strip()
        workspace_header = request.headers.get("x-workspace-id", "").strip()

        if not raw_key or not workspace_header:
            resp = _err(401, "Missing x-api-key or x-workspace-id header", request_id)
            await resp(scope, receive, send)
            return

        try:
            workspace_id = uuid.UUID(workspace_header)
        except ValueError:
            resp = _err(401, "Invalid x-workspace-id format", request_id)
            await resp(scope, receive, send)
            return

        key_hash = _sha256(raw_key)
        payload = await _resolve_api_key(key_hash, workspace_id)

        if payload is None:
            resp = _err(401, "Invalid API key", request_id)
            await resp(scope, receive, send)
            return

        if not payload["key_is_active"]:
            resp = _err(401, "API key has been revoked", request_id)
            await resp(scope, receive, send)
            return

        if str(payload["tenant_id"]) != str(workspace_id):
            resp = _err(401, "API key does not belong to the specified workspace", request_id)
            await resp(scope, receive, send)
            return

        if payload["tenant_status"] != "active":
            resp = _err(401, "Workspace is not active", request_id)
            await resp(scope, receive, send)
            return

        # Attach context to request.state for downstream handlers
        request.state.api_key_id = uuid.UUID(payload["api_key_id"])
        request.state.workspace_id = workspace_id
        # Full Tenant object is not attached here to keep middleware < 5 ms;
        # endpoints that need it can load via get_db() + tenant_id from state.

        await self._call_with_request_id(scope, receive, send, request_id)

    async def _call_with_request_id(self, scope, receive, send, request_id: str):
        """Wrap send to inject X-Request-ID into every response."""
        async def send_with_header(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_header)
