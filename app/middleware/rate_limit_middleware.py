"""
Global sliding-window rate limiter middleware.

Enforces API_RATE_LIMIT requests per API_RATE_WINDOW seconds per authenticated
identity using Redis sorted sets (true sliding window, multi-pod safe).

Identity resolution (first match wins):
  api_key auth  → SHA-256 of raw x-api-key header
  jwt auth      → "jwt:{user_id}"
  fallback      → "ip:{client_host}"

Redis key: rate_limit:{identity}
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import redis.asyncio as aioredis
from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.config import settings
from app.core.logger import logger
from app.core.request_auth import AUTH_METHOD_API_KEY, AUTH_METHOD_JWT
from app.middleware.request_id_middleware import get_request_id

# Built-in /docs, /redoc, /openapi.json disabled in app factory; rate limit applies only to /api/v1/*.
# Custom docs at /api/docs are outside that scope (see app/routers/api_docs.py).
_SKIP_EXACT = {
    "/",
    "/health",
    "/api/v1/health",
    "/api/v1/tenants/create",
}

# Prefix skips for webhooks, voice/streaming, and public checkout — not user auth routes.
_SKIP_PREFIXES = (
    "/api/v1/api-keys/",
    "/api/v1/plans/public",
    "/api/v1/tenants/start-credit-checkout-session",
    "/api/v1/auth/",
    "/api/v1/billing/webhook",
    "/api/v1/voice/",
    "/api/v1/stream/",
)

# Unauthenticated POST endpoints targeted by bots (per-path, per-IP stricter limit).
_AUTH_SENSITIVE_POST_PATHS: frozenset[str] = frozenset({
    "/api/v1/users/register",
    "/api/v1/users/forgot-password",
    "/api/v1/users/reset-password",
    "/api/v1/users/refresh",
    "/api/v1/accept-invite/accept-invite",
})

# Login/google login use enforce_login_rate_limit() on the route (same Redis helper).
_LOGIN_POST_PATHS: frozenset[str] = frozenset({
    "/api/v1/users/login",
    "/api/v1/users/login/google",
})

# Public, unauthenticated Web SDK token endpoint — strict per-IP cap (20/min)
# since there's no API key/JWT identity to bucket on otherwise.
_PUBLIC_TOKEN_POST_PATHS: frozenset[str] = frozenset({
    "/api/v1/sdk/public-call-token",
})


def _should_skip(path: str) -> bool:
    if path in _SKIP_EXACT:
        return True
    return any(path.startswith(p) for p in _SKIP_PREFIXES)


def _client_host(scope: Scope) -> str:
    client = scope.get("client")
    return client[0] if client else "unknown"


def _is_auth_sensitive_post(scope: Scope) -> bool:
    if scope.get("method") != "POST":
        return False
    path = scope.get("path", "")
    return path in _AUTH_SENSITIVE_POST_PATHS


def _is_public_token_post(scope: Scope) -> bool:
    if scope.get("method") != "POST":
        return False
    path = scope.get("path", "")
    return path in _PUBLIC_TOKEN_POST_PATHS


def _sha256(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


_redis: Optional[aioredis.Redis] = None


def _get_redis() -> Optional[aioredis.Redis]:
    global _redis
    if _redis is None:
        try:
            _redis = aioredis.from_url(
                settings.REDIS_URL, encoding="utf-8", decode_responses=True
            )
        except Exception as exc:
            logger.warning("Rate limit middleware: Redis unavailable (%s) — skipping", exc)
    return _redis


async def close_rate_limit_middleware() -> None:
    global _redis
    if _redis is not None:
        try:
            await _redis.aclose()
        except Exception as exc:
            logger.warning("Rate limit Redis close failed: %s", exc)
        _redis = None


def _identity_key(scope: Scope) -> str:
    state = scope.get("state", {})

    auth_method = getattr(state, "auth_method", None)
    if auth_method == AUTH_METHOD_API_KEY:
        headers = dict(scope.get("headers", []))
        raw_key = headers.get(b"x-api-key", b"").decode("latin-1").strip()
        return _sha256(raw_key) if raw_key else "unknown"

    if auth_method == AUTH_METHOD_JWT:
        user_id = getattr(state, "user_id", None)
        return f"jwt:{user_id}" if user_id else "jwt:unknown"

    return f"ip:{_client_host(scope)}"


async def _check_rate_limit(key: str, limit: int, window: int) -> tuple[bool, float]:
    """
    Sliding window check using a Redis sorted set.

    Returns (allowed, retry_after_timestamp).
    retry_after_timestamp is epoch seconds when the oldest slot frees.
    """
    r = _get_redis()
    if r is None:
        return True, 0.0

    now = time.time()
    window_start = now - window
    redis_key = f"rate_limit:{key}"

    try:
        pipe = r.pipeline()
        pipe.zremrangebyscore(redis_key, "-inf", window_start)
        pipe.zadd(redis_key, {f"{now}:{uuid.uuid4().hex}": now})
        pipe.zcard(redis_key)
        pipe.zrange(redis_key, 0, 0, withscores=True)
        pipe.expire(redis_key, window)
        results = await pipe.execute()

        count = results[2]
        oldest = results[3]

        if count > limit:
            # Undo the zadd we just did — remove the entry we added
            await r.zremrangebyscore(redis_key, now, now + 0.001)
            retry_after = (oldest[0][1] + window) if oldest else (now + window)
            return False, retry_after

        return True, 0.0

    except Exception as exc:
        logger.warning("Rate limit Redis check failed: %s — allowing request", exc)
        return True, 0.0


def build_rate_limit_error(retry_after_ts: float) -> dict[str, str]:
    """Structured rate-limit error body (inner ``error`` object, without requestId)."""
    retry_dt = datetime.fromtimestamp(retry_after_ts, tz=timezone.utc)
    retry_iso = retry_dt.isoformat().replace("+00:00", "Z")
    return {
        "code": "rate_limit_exceeded",
        "message": "Too many requests. Please retry after the time indicated.",
        "retryAfter": retry_iso,
    }


def _rate_limited_response(retry_after_ts: float) -> bytes:
    return json.dumps({"error": build_rate_limit_error(retry_after_ts)}).encode()


class RateLimitMiddleware:
    """Global sliding-window rate limiter — runs after ApiKeyMiddleware resolves auth."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if scope.get("method") == "OPTIONS" or _should_skip(path):
            await self.app(scope, receive, send)
            return

        if not path.startswith("/api/v1/"):
            await self.app(scope, receive, send)
            return

        if not settings.RATE_LIMIT_ENABLED:
            await self.app(scope, receive, send)
            return

        # Stricter per-IP limit on auth-adjacent POST routes (register, reset, etc.).
        if _is_auth_sensitive_post(scope):
            auth_key = f"auth:{path}:{_client_host(scope)}"
            allowed, retry_after = await _check_rate_limit(
                auth_key, settings.LOGIN_RATE_LIMIT, settings.LOGIN_RATE_WINDOW
            )
            if not allowed:
                await self._send_rate_limited(scope, receive, send, retry_after)
                return

        # Public Web SDK token endpoint — no auth, so this is the only per-IP cap.
        if _is_public_token_post(scope):
            token_key = f"public_token:{path}:{_client_host(scope)}"
            allowed, retry_after = await _check_rate_limit(
                token_key, settings.PUBLIC_TOKEN_RATE_LIMIT, settings.PUBLIC_TOKEN_RATE_WINDOW
            )
            if not allowed:
                await self._send_rate_limited(scope, receive, send, retry_after)
                return

        # Login routes rely on enforce_login_rate_limit() Depends — skip middleware auth bucket.
        if scope.get("method") == "POST" and path in _LOGIN_POST_PATHS:
            await self.app(scope, receive, send)
            return

        key = _identity_key(scope)
        allowed, retry_after = await _check_rate_limit(
            key, settings.API_RATE_LIMIT, settings.API_RATE_WINDOW
        )

        if not allowed:
            await self._send_rate_limited(scope, receive, send, retry_after)
            return

        await self.app(scope, receive, send)

    async def _send_rate_limited(
        self, scope: Scope, receive: Receive, send: Send, retry_after: float
    ) -> None:
        from starlette.requests import Request

        request = Request(scope, receive)
        request_id = get_request_id(request)
        body = _rate_limited_response(retry_after)
        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"x-request-id", request_id.encode()],
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
