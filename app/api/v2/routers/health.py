from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.async_session import get_db as get_async_db

router = APIRouter(tags=["v2"])


@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_async_db)) -> dict:
    try:
        await db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception:
        db_status = "error"

    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "version": settings.APP_VERSION,
        "service": "fastapi-v2",
        "db": db_status,
    }
