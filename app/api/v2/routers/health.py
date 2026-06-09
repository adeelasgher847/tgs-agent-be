from fastapi import APIRouter

from app.core.config import settings

router = APIRouter(tags=["v2"])


@router.get("/health")
async def health_check() -> dict:
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "service": "fastapi-v2",
    }
