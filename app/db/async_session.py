from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.core.logger import logger
from app.db.async_url import database_url_to_async

_async_engine = None
_AsyncSessionLocal: async_sessionmaker | None = None


def init_async_db() -> None:
    global _async_engine, _AsyncSessionLocal
    _async_engine = create_async_engine(
        database_url_to_async(settings.DATABASE_URL),
        pool_pre_ping=True,
    )
    _AsyncSessionLocal = async_sessionmaker(
        _async_engine, class_=AsyncSession, expire_on_commit=False
    )
    logger.info("Async DB pool initialized")


async def dispose_async_db() -> None:
    global _async_engine, _AsyncSessionLocal
    if _async_engine is not None:
        await _async_engine.dispose()
        _async_engine = None
        _AsyncSessionLocal = None
        logger.info("Async DB pool disposed")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    if _AsyncSessionLocal is None:
        raise RuntimeError("Async DB not initialized — call init_async_db() at startup")
    async with _AsyncSessionLocal() as session:
        yield session
