from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse

from app.api.api_v1.api import api_router
from app.routers.health import router as health_router
from app.schemas.base import SuccessResponse
from app.utils.response import create_success_response
from app.utils.rate_limiter import init_rate_limiter

from app.core.config import settings
from app.core.logger import setup_logging, logger
from app.core.secret_manager import get_rime_api_key
from app.core.shutdown import graceful_shutdown
from app.core.exception_handlers import register_exception_handlers
from app.middleware.api_key_middleware import ApiKeyMiddleware
from app.middleware.body_limit_middleware import BodyLimitMiddleware
from app.middleware.pii_logging_middleware import PiiLoggingMiddleware
from app.middleware.rate_limit_middleware import RateLimitMiddleware
from app.middleware.request_id_middleware import RequestIdMiddleware

# ---------------------------------------------------------------------------
# Logging — must be first so every subsequent import uses the configured logger.
# ---------------------------------------------------------------------------
setup_logging()

# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown lifecycle (uvicorn SIGTERM triggers shutdown).
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- startup ----
    logger.info("Application startup initiated")
    try:
        await init_rate_limiter()
        logger.info("Rate limiter initialized successfully")
    except Exception as exc:
        logger.warning("Rate limiter initialization failed: %s — continuing without rate limiting", exc)

    try:
        get_rime_api_key()
        logger.info("Rime TTS API key configured")
    except (ValueError, RuntimeError) as exc:
        logger.error("Rime TTS misconfigured: %s", exc)
        raise

    yield

    # ---- shutdown (SIGTERM / reload) — drain connections via uvicorn, then cleanup. ----
    await graceful_shutdown()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    openapi_version="3.0.3",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# OpenAPI schema patch: convert contentMediaType → format=binary for Swagger.
# ---------------------------------------------------------------------------
def _patch_binary_formats(schema_obj):
    if isinstance(schema_obj, dict):
        if (
            schema_obj.get("type") == "string"
            and schema_obj.get("contentMediaType") == "application/octet-stream"
        ):
            schema_obj["format"] = "binary"
            schema_obj.pop("contentMediaType", None)
        for v in schema_obj.values():
            _patch_binary_formats(v)
    elif isinstance(schema_obj, list):
        for item in schema_obj:
            _patch_binary_formats(item)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title or "FastAPI",
        version=app.version or "0.1.0",
        description=app.description,
        routes=app.routes,
    )
    _patch_binary_formats(openapi_schema)
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# ---------------------------------------------------------------------------
# Exception handlers — registered before middleware so they fire on errors
# that bubble out of the middleware stack.
# ---------------------------------------------------------------------------
register_exception_handlers(app)

# ---------------------------------------------------------------------------
# Middleware stack (add_middleware is LIFO — LAST added = OUTERMOST on request).
#
# Incoming (outer → inner):
#   CORS → RequestId → BodyLimit → PiiLogging → ApiKey → RateLimit → handler
#
# CORS must be outermost so browser OPTIONS preflight gets Allow-Origin headers
# before ApiKey can return 401. M2M clients (no browser) skip preflight entirely.
# RateLimit runs after ApiKey so auth identity is resolved before rate-keying.
# ---------------------------------------------------------------------------

_allowed_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]

# 1. Innermost — sliding-window rate limiter (auth identity already resolved).
app.add_middleware(RateLimitMiddleware)

# 2. API key / JWT auth (actual requests only; OPTIONS passes through).
app.add_middleware(ApiKeyMiddleware)

# 3. PII-safe request logging
app.add_middleware(PiiLoggingMiddleware)

# 4. Body size limit (10 MB)
app.add_middleware(BodyLimitMiddleware)

# 5. Request ID — nanoid on request.state before auth errors are built.
app.add_middleware(RequestIdMiddleware)

# 6. Outermost — CORS handles preflight and adds headers to all responses (incl. 401).
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_model=SuccessResponse[dict])
def read_root():
    return create_success_response(
        {"message": "Welcome to the Multi-Tenant SaaS Voice Agent Backend!"},
        "API is running successfully",
    )


app.include_router(api_router, prefix="/api/v1")
app.include_router(health_router)
