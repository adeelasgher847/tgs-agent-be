from datetime import datetime, timezone

from fastapi import APIRouter

from app.core.config import settings
from app.services.livekit_service import livekit_service

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    try:
        livekit_status = await livekit_service.health_check()
    except Exception:
        livekit_status = "degraded"

    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "livekit": livekit_status,
    }
