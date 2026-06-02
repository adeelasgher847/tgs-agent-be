"""Graceful application shutdown — release connections and background resources."""

from __future__ import annotations

from app.core.logger import logger
from app.utils.rate_limiter import close_rate_limiter


async def graceful_shutdown() -> None:
    """
    Run on application shutdown (FastAPI lifespan / uvicorn SIGTERM).

    Closes rate-limiter Redis and auth-middleware pools so in-flight requests
    can finish without leaving dangling connections.
    """
    logger.info("Graceful shutdown started")

    try:
        await close_rate_limiter()
        logger.info("Rate limiter closed")
    except Exception as exc:
        logger.error("Rate limiter cleanup failed: %s", exc)

    try:
        from app.middleware.api_key_middleware import close_auth_middleware_resources

        await close_auth_middleware_resources()
        logger.info("Auth middleware resources closed")
    except Exception as exc:
        logger.error("Auth middleware cleanup failed: %s", exc)

    logger.info("Graceful shutdown complete")
