"""
Public health endpoint — GET /api/v1/health.

No API key or JWT required (see the skip lists in
app/middleware/api_key_middleware.py and app/middleware/rate_limit_middleware.py).

Betterstack (https://betterstack.com/) polls this endpoint every 30 seconds
expecting an HTTP 200, and uses it to drive the public status page served at
status.yourdomain.com. Configure BETTERSTACK_BADGE_URL to surface that page's
embeddable badge in the response.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from app.core.config import settings

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "betterstack_badge": settings.BETTERSTACK_BADGE_URL,
    }
