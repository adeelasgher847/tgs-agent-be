from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from app.api.api_v1.api import api_router
from app.api.v2.api import v2_router
from app.db.async_session import dispose_async_db, init_async_db
from app.routers.api_docs import router as api_docs_router
from app.routers.health import router as health_router
from app.schemas.base import SuccessResponse
from app.utils.response import create_success_response
from app.utils.arq_pool import init_arq_pool
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
        init_async_db()
        logger.info("Async DB pool initialized")
    except Exception as exc:
        logger.critical("Failed to initialize async DB pool: %s", exc, exc_info=True)
        raise
    try:
        await init_rate_limiter()
        logger.info("Rate limiter initialized successfully")
    except Exception as exc:
        logger.warning("Rate limiter initialization failed: %s — continuing without rate limiting", exc)

    try:
        await init_arq_pool()
    except Exception as exc:
        logger.warning("ARQ pool startup failed: %s — batch enqueue will use per-request pool", exc)

    try:
        get_rime_api_key()
        logger.info("Rime TTS API key configured")
    except (ValueError, RuntimeError) as exc:
        logger.error("Rime TTS misconfigured: %s", exc)
        raise

    if settings.LIVEKIT_ENABLED:
        from app.core.secret_manager import get_livekit_credentials

        try:
            get_livekit_credentials()
            logger.info("LiveKit credentials configured")
        except (ValueError, RuntimeError) as exc:
            env = settings.ENVIRONMENT.lower()
            if env in ("staging", "production"):
                logger.error("LiveKit credentials missing in %s: %s", env, exc)
                raise
            else:
                logger.warning(
                    "LiveKit credentials not configured: %s — set LIVEKIT_ENABLED=false "
                    "or add LIVEKIT_URL/LIVEKIT_API_KEY/LIVEKIT_API_SECRET to .env",
                    exc,
                )

    if settings.API_DOCS_ENABLED:
        if settings.API_DOCS_USERNAME and settings.API_DOCS_PASSWORD:
            logger.info(
                "API docs enabled at /api/docs (HTTP Basic user=%s)",
                settings.API_DOCS_USERNAME,
            )
        else:
            logger.warning(
                "API docs enabled but API_DOCS_USERNAME/API_DOCS_PASSWORD are empty — "
                "/api/docs will return 503 until set; restart server after updating .env"
            )

    yield

    # ---- shutdown (SIGTERM / reload) — drain connections via uvicorn, then cleanup. ----
    await dispose_async_db()
    await graceful_shutdown()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
# Built-in /docs, /redoc, and /openapi.json are disabled (public spec is served at
# GET /api/docs with HTTP Basic). openapi_url=None removes only those HTTP routes;
# programmatic schema generation still works via custom_openapi() / app.openapi()
# (contract tests, scripts/export_openapi.py).
app = FastAPI(
    title="Happy Assist Ai",
    description="Multi-tenant SaaS Voice Agent Backend",
    openapi_version="3.0.3",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
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


_WORKSPACE_API_KEY_SECURITY = [{"ApiKeyAuth": [], "WorkspaceId": []}]


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

    # ---- security schemes ----
    components = openapi_schema.setdefault("components", {})
    schemes = components.setdefault("securitySchemes", {})
    # JWT bearer already auto-generated by FastAPI; keep it and add API key schemes.
    schemes["ApiKeyAuth"] = {
        "type": "apiKey",
        "in": "header",
        "name": "x-api-key",
        "description": "API key for machine-to-machine authentication. Required alongside x-workspace-id.",
    }
    schemes["WorkspaceId"] = {
        "type": "apiKey",
        "in": "header",
        "name": "x-workspace-id",
        "description": "Workspace (tenant) UUID. Required alongside x-api-key.",
    }

    # ---- apply security to workspace CRUD paths ----
    paths = openapi_schema.get("paths", {})
    for path, path_item in paths.items():
        if path.startswith("/api/v1/workspace"):
            for _method, operation in path_item.items():
                if isinstance(operation, dict):
                    operation["security"] = _WORKSPACE_API_KEY_SECURITY

    openapi_schema["openapi"] = "3.0.3"
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

# 4. Body size limit — 52 MB to allow KB file uploads (max 50 MB per endpoint)
app.add_middleware(BodyLimitMiddleware, max_bytes=52 * 1024 * 1024)

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


app.include_router(api_docs_router, prefix="/api")
app.include_router(api_router, prefix="/api/v1")
app.include_router(health_router)
app.include_router(v2_router, prefix="/api/v2")


# ---------------------------------------------------------------------------
# v2 Swagger — filtered to /api/v2/ routes only.
# ---------------------------------------------------------------------------
@app.get("/api/v2/openapi.json", include_in_schema=False)
async def v2_openapi_schema() -> dict:
    from fastapi.openapi.utils import get_openapi

    v2_routes = [r for r in app.routes if getattr(r, "path", "").startswith("/api/v2/")]
    return get_openapi(
        title="TGS API v2",
        version=settings.APP_VERSION,
        routes=v2_routes,
    )


@app.get("/api/v2/docs", include_in_schema=False)
async def v2_swagger_ui():
    from fastapi.openapi.docs import get_swagger_ui_html
    from fastapi.responses import HTMLResponse

    html = get_swagger_ui_html(
        openapi_url="/api/v2/openapi.json",
        title="TGS API v2 — Docs",
    )
    return HTMLResponse(html.body)
