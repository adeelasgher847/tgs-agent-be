"""
Dual authentication middleware for /api/v1/* and /api/v2/* routes.

Auth resolution order (first match wins):
  1. Public / skip paths → pass through (no auth required)
  2. Valid ``x-api-key`` + ``x-workspace-id`` → API key auth
  3. Valid ``Authorization: Bearer <JWT>`` → dashboard user auth
  4. Otherwise → HTTP 401

On success the resolved workspace (tenant) is attached to the request:
  ``request.state.workspace`` — immutable :class:`~app.core.workspace.Workspace`
  ``request.state.workspace_id`` — same as ``workspace.id`` (convenience)
  ``request.state.auth_method`` — ``"api_key"`` or ``"jwt"``

API key lookups are cached in Redis (TTL 60 s). JWT workspace loads use a
separate Redis cache keyed by workspace id.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Optional

import redis.asyncio as aioredis
from fastapi import Request
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from app.core.auth_tokens import extract_bearer_token, resolve_jwt_auth
from app.core.config import settings
from app.core.error_responses import build_api_error_payload
from app.core.logger import logger
from app.core.request_auth import AUTH_METHOD_API_KEY, AUTH_METHOD_JWT
from app.core.workspace import Workspace
from app.middleware.request_id_middleware import get_request_id
from app.models.api_key import Apikey
from app.models.tenant import Tenant

_redis: Optional[aioredis.Redis] = None

CACHE_TTL = 60  # seconds

_async_engine = None
_AsyncSessionLocal: Optional[sessionmaker] = None

_SKIP_EXACT = {
    "/",
    "/api/v1/tenants/create",
}

_SKIP_PREFIXES = (
    "/api/v1/users/",
    "/api/v1/api-keys/",
    "/api/v1/accept-invite",
    "/api/v1/plans/public",
    "/api/v1/tenants/start-credit-checkout-session",
    "/api/v1/auth/",
    "/api/v1/billing/webhook",
    "/api/v1/voice/",
    "/api/v1/stream/",
    # Public Web SDK endpoints — security enforced via flow.public_access +
    # allowed_domains Origin check inside the handler, not API credentials.
    "/api/v1/sdk/",
    "/api/v1/integrations/hubspot/callback",
    "/health",
    # v2 public endpoints — no auth required
    "/api/v2/health",
    "/api/v2/docs",
    "/api/v2/openapi.json",
)

def _get_redis() -> Optional[aioredis.Redis]:
    global _redis
    if _redis is None:
        try:
            _redis = aioredis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Auth middleware: Redis unavailable (%s) — cache disabled", exc)
    return _redis


async def close_auth_middleware_resources() -> None:
    """Release Redis and async SQLAlchemy engine created for auth lookups."""
    global _redis, _async_engine, _AsyncSessionLocal

    if _redis is not None:
        try:
            await _redis.aclose()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Auth middleware Redis close failed: %s", exc)
        _redis = None

    if _async_engine is not None:
        try:
            await _async_engine.dispose()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Auth middleware engine dispose failed: %s", exc)
        _async_engine = None
        _AsyncSessionLocal = None


def _get_async_session() -> AsyncSession:
    global _async_engine, _AsyncSessionLocal
    if _async_engine is None:
        from sqlalchemy.ext.asyncio import create_async_engine as _cae

        from app.db.async_url import database_url_to_async

        _async_engine = _cae(
            database_url_to_async(settings.DATABASE_URL),
            pool_pre_ping=True,
        )
        _AsyncSessionLocal = sessionmaker(
            _async_engine, class_=AsyncSession, expire_on_commit=False
        )
    return _AsyncSessionLocal()


def _should_skip(path: str) -> bool:
    if path in _SKIP_EXACT:
        return True
    return any(path.startswith(prefix) for prefix in _SKIP_PREFIXES)


def _sha256(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def _unauthorized(request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content=build_api_error_payload(
            401,
            "Invalid or missing API key",
            error_code="unauthorized",
            request_id=request_id,
        ),
        headers={"X-Request-ID": request_id},
    )


def _attach_workspace_context(
    request: Request,
    *,
    workspace: Workspace,
    auth_method: str,
    user_id: Optional[uuid.UUID] = None,
    api_key_id: Optional[uuid.UUID] = None,
    api_key_prefix: Optional[str] = None,
) -> None:
    request.state.workspace = workspace
    request.state.workspace_id = workspace.id
    request.state.auth_method = auth_method
    request.state.user_id = user_id
    request.state.api_key_id = api_key_id
    request.state.api_key_prefix = api_key_prefix


def _workspace_from_api_key_payload(payload: dict) -> Optional[Workspace]:
    workspace_data = payload.get("workspace")
    if workspace_data is None:
        return None
    try:
        return Workspace.from_mapping(workspace_data)
    except (KeyError, TypeError, ValueError):
        return None


def _api_key_payload_valid(payload: dict, workspace_id: uuid.UUID) -> bool:
    workspace = _workspace_from_api_key_payload(payload)
    if workspace is None:
        return False
    return (
        payload.get("key_is_active") is True
        and workspace.id == workspace_id
        and workspace.is_active
    )


# ---------------------------------------------------------------------------
# Redis: API key auth cache
# ---------------------------------------------------------------------------

def _apikey_cache_key(key_hash: str, workspace_id: uuid.UUID) -> str:
    return f"apikey:{key_hash}:{workspace_id}"


async def _apikey_cache_get(key_hash: str, workspace_id: uuid.UUID) -> Optional[dict]:
    r = _get_redis()
    if r is None:
        return None
    try:
        raw = await r.get(_apikey_cache_key(key_hash, workspace_id))
        return json.loads(raw) if raw else None
    except Exception as exc:
        logger.debug("Redis apikey get failed: %s", exc)
        return None


async def _apikey_cache_set(key_hash: str, workspace_id: uuid.UUID, payload: dict) -> None:
    r = _get_redis()
    if r is None:
        return
    try:
        await r.setex(
            _apikey_cache_key(key_hash, workspace_id),
            CACHE_TTL,
            json.dumps(payload),
        )
    except Exception as exc:
        logger.debug("Redis apikey set failed: %s", exc)


async def _apikey_cache_delete(key_hash: str, workspace_id: uuid.UUID) -> None:
    r = _get_redis()
    if r is None:
        return
    try:
        await r.delete(_apikey_cache_key(key_hash, workspace_id))
    except Exception as exc:
        logger.debug("Redis apikey delete failed: %s", exc)


async def invalidate_api_key_cache(raw_key: str, workspace_id: uuid.UUID) -> None:
    await _apikey_cache_delete(_sha256(raw_key), workspace_id)


async def invalidate_api_key_cache_by_hash(key_hash: str, workspace_id: uuid.UUID) -> None:
    await _apikey_cache_delete(key_hash, workspace_id)


# ---------------------------------------------------------------------------
# Redis: workspace (tenant) cache — used for JWT auth path
# ---------------------------------------------------------------------------

def _workspace_cache_key(workspace_id: uuid.UUID) -> str:
    return f"workspace:{workspace_id}"


async def _workspace_cache_get(workspace_id: uuid.UUID) -> Optional[Workspace]:
    r = _get_redis()
    if r is None:
        return None
    try:
        raw = await r.get(_workspace_cache_key(workspace_id))
        if not raw:
            return None
        return Workspace.from_mapping(json.loads(raw))
    except Exception as exc:
        logger.debug("Redis workspace get failed: %s", exc)
        return None


async def _workspace_cache_set(workspace: Workspace) -> None:
    r = _get_redis()
    if r is None:
        return
    try:
        await r.setex(
            _workspace_cache_key(workspace.id),
            CACHE_TTL,
            json.dumps(workspace.to_cache_dict()),
        )
    except Exception as exc:
        logger.debug("Redis workspace set failed: %s", exc)


async def invalidate_workspace_cache(workspace_id: uuid.UUID) -> None:
    """Call when tenant/workspace fields change (billing, status, etc.)."""
    r = _get_redis()
    if r is None:
        return
    try:
        await r.delete(_workspace_cache_key(workspace_id))
    except Exception as exc:
        logger.debug("Redis workspace delete failed: %s", exc)


async def _load_workspace(workspace_id: uuid.UUID) -> Optional[Workspace]:
    cached = await _workspace_cache_get(workspace_id)
    if cached is not None:
        return cached

    try:
        async with _get_async_session() as session:
            tenant = await session.get(Tenant, workspace_id)
    except Exception as exc:
        logger.error("Auth middleware workspace load failed: %s", exc, exc_info=True)
        return None

    if tenant is None:
        return None

    workspace = Workspace.from_tenant(tenant)
    await _workspace_cache_set(workspace)
    return workspace


def _build_api_key_payload(api_key_obj: Apikey, tenant_obj: Tenant) -> dict[str, Any]:
    workspace = Workspace.from_tenant(tenant_obj)
    return {
        "api_key_id": str(api_key_obj.id),
        "tenant_id": str(tenant_obj.id),
        "key_is_active": api_key_obj.is_active,
        "workspace": workspace.to_cache_dict(),
    }


async def _resolve_api_key(key_hash: str, workspace_id: uuid.UUID) -> Optional[dict]:
    cached = await _apikey_cache_get(key_hash, workspace_id)
    if cached is not None and cached.get("workspace") is not None:
        return cached

    try:
        async with _get_async_session() as session:
            result = await session.execute(
                select(Apikey, Tenant)
                .join(Tenant, Apikey.tenant_id == Tenant.id)
                .where(
                    Apikey.key_hash == key_hash,
                    Apikey.tenant_id == workspace_id,
                )
            )
            row = result.first()
    except Exception as exc:
        logger.error("Auth middleware DB lookup failed: %s", exc, exc_info=True)
        return None

    if row is None:
        return None

    api_key_obj, tenant_obj = row
    payload = _build_api_key_payload(api_key_obj, tenant_obj)
    await _apikey_cache_set(key_hash, workspace_id, payload)
    return payload


async def _try_api_key_auth(request: Request) -> bool:
    raw_key = request.headers.get("x-api-key", "").strip()
    workspace_header = request.headers.get("x-workspace-id", "").strip()
    if not raw_key or not workspace_header:
        return False

    try:
        workspace_id = uuid.UUID(workspace_header)
    except ValueError:
        return False

    payload = await _resolve_api_key(_sha256(raw_key), workspace_id)
    if payload is None or not _api_key_payload_valid(payload, workspace_id):
        return False

    workspace = _workspace_from_api_key_payload(payload)
    if workspace is None:
        return False

    _attach_workspace_context(
        request,
        workspace=workspace,
        auth_method=AUTH_METHOD_API_KEY,
        api_key_id=uuid.UUID(payload["api_key_id"]),
        api_key_prefix=raw_key[:8],
    )
    return True


async def _try_jwt_auth(request: Request) -> bool:
    token = extract_bearer_token(request.headers.get("authorization"))
    if not token:
        return False

    jwt_ctx = resolve_jwt_auth(token)
    if jwt_ctx is None:
        return False

    workspace = await _load_workspace(jwt_ctx["workspace_id"])
    if workspace is None:
        return False

    _attach_workspace_context(
        request,
        workspace=workspace,
        auth_method=AUTH_METHOD_JWT,
        user_id=jwt_ctx["user_id"],
    )
    return True


class ApiKeyMiddleware:
    """Enforces API-key OR JWT auth on protected /api/v1 and /api/v2 routes."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        request_id = get_request_id(request)

        # Browser CORS preflight — no auth headers; must reach CORSMiddleware.
        if request.method == "OPTIONS":
            await self.app(scope, receive, send)
            return

        path = request.url.path
        is_protected_api = path.startswith("/api/v1/") or path.startswith("/api/v2/")
        if not is_protected_api or _should_skip(path):
            await self.app(scope, receive, send)
            return

        if await _try_api_key_auth(request) or await _try_jwt_auth(request):
            await self.app(scope, receive, send)
            return

        await _unauthorized(request_id)(scope, receive, send)
