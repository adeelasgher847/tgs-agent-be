from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi


def create_app() -> FastAPI:
    """
    Build and return the FastAPI application.

    No database connections, external service calls, or blocking I/O occur here.
    All runtime resources (DB pool, Redis, ARQ, secrets) are initialised inside
    the lifespan context manager, which only runs when the server starts.
    """
    # Logging must be configured before any other import that uses the logger.
    from app.core.logger import setup_logging, logger
    setup_logging()

    from app.api.api_v1.api import api_router
    from app.api.v2.api import v2_router
    from app.routers.sso_auth import router as sso_auth_router
    from app.db.async_session import dispose_async_db, init_async_db
    from app.routers.api_docs import router as api_docs_router
    from app.routers.health import router as health_router
    from app.schemas.base import SuccessResponse
    from app.utils.response import create_success_response
    from app.utils.arq_pool import init_arq_pool
    from app.utils.rate_limiter import init_rate_limiter
    from app.core.config import settings
    from app.core.secret_manager import get_rime_api_key
    from app.core.shutdown import graceful_shutdown
    from app.core.exception_handlers import register_exception_handlers
    from app.middleware.api_key_middleware import ApiKeyMiddleware
    from app.middleware.body_limit_middleware import BodyLimitMiddleware
    from app.middleware.pii_logging_middleware import PiiLoggingMiddleware
    from app.middleware.public_sdk_cors_middleware import PublicSdkCorsMiddleware
    from app.middleware.rate_limit_middleware import RateLimitMiddleware
    from app.middleware.request_id_middleware import RequestIdMiddleware

    # -------------------------------------------------------------------------
    # Lifespan — all runtime I/O lives here, never at module level.
    # -------------------------------------------------------------------------
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # ---- startup ----
        logger.info("Application startup initiated")

        from app.core.observability import setup_tracing
        setup_tracing(app)

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
            # Rime is always seeded as an available platform provider for the
            # multi-tenant SaaS deployment (any tenant's agent can pick it),
            # so cloud environments enforce this eagerly. On-premise/BYO
            # deployments pick exactly one TTS_PROVIDER and may not have a
            # Rime account at all — only hard-fail there if Rime is actually
            # the configured provider. Mirrors the LiveKit check below.
            env = settings.ENVIRONMENT.lower()
            if env in ("staging", "production") and settings.TTS_PROVIDER == "rime":
                logger.error("Rime TTS misconfigured: %s", exc)
                raise
            else:
                logger.warning("Rime TTS not configured: %s — fine if TTS_PROVIDER is not 'rime'", exc)

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

        # ---- shutdown (SIGTERM / reload) ----
        await dispose_async_db()
        await graceful_shutdown()

    # -------------------------------------------------------------------------
    # FastAPI instance
    # Built-in /docs, /redoc, /openapi.json disabled; spec served at /api/docs
    # with HTTP Basic. Programmatic schema generation (app.openapi()) still works.
    # -------------------------------------------------------------------------
    _app = FastAPI(
        title="Happy Assist Ai",
        description="Multi-tenant SaaS Voice Agent Backend",
        openapi_version="3.0.3",
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # -------------------------------------------------------------------------
    # OpenAPI schema: binary format patch + security schemes
    # -------------------------------------------------------------------------
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

    _WORKSPACE_API_KEY_SECURITY = [
        {"ApiKeyAuth": [], "WorkspaceId": []},
        {"HTTPBearer": []}
    ]

    def custom_openapi():
        if _app.openapi_schema:
            return _app.openapi_schema
        openapi_schema = get_openapi(
            title=_app.title or "FastAPI",
            version=_app.version or "0.1.0",
            description=_app.description,
            routes=_app.routes,
        )
        _patch_binary_formats(openapi_schema)

        components = openapi_schema.setdefault("components", {})
        schemes = components.setdefault("securitySchemes", {})
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
        schemes["HTTPBearer"] = {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Enter your JWT token.",
        }

        paths = openapi_schema.get("paths", {})
        for path, path_item in paths.items():
            if path.startswith("/api/v1/workspace"):
                for _method, operation in path_item.items():
                    if isinstance(operation, dict):
                        operation["security"] = _WORKSPACE_API_KEY_SECURITY

        openapi_schema["openapi"] = "3.0.3"
        _app.openapi_schema = openapi_schema
        return _app.openapi_schema

    _app.openapi = custom_openapi

    # -------------------------------------------------------------------------
    # Exception handlers — before middleware so they fire on middleware errors.
    # -------------------------------------------------------------------------
    register_exception_handlers(_app)

    # -------------------------------------------------------------------------
    # Middleware stack (add_middleware is LIFO — LAST added = OUTERMOST on request)
    #
    # Incoming (outer → inner):
    #   PublicSdkCors → CORS → RequestId → BodyLimit → PiiLogging → ApiKey → RateLimit → handler
    #
    # PublicSdkCors only acts on /api/v1/sdk/public-call-token (reflects Origin
    # dynamically for the allowed_domains whitelist); every other path passes
    # through it untouched and is governed by the static CORS config below.
    # -------------------------------------------------------------------------
    _allowed_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]

    _app.add_middleware(RateLimitMiddleware)
    _app.add_middleware(ApiKeyMiddleware)
    _app.add_middleware(PiiLoggingMiddleware)
    _app.add_middleware(BodyLimitMiddleware, max_bytes=52 * 1024 * 1024)
    _app.add_middleware(RequestIdMiddleware)
    _app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    _app.add_middleware(PublicSdkCorsMiddleware)

    # -------------------------------------------------------------------------
    # Routes
    # -------------------------------------------------------------------------
    @_app.get("/", response_model=SuccessResponse[dict])
    def read_root():
        return create_success_response(
            {"message": "Welcome to the Multi-Tenant SaaS Voice Agent Backend!"},
            "API is running successfully",
        )

    _app.include_router(api_docs_router, prefix="/api")
    _app.include_router(api_router, prefix="/api/v1")
    _app.include_router(health_router)
    _app.include_router(v2_router, prefix="/api/v2")
    _app.include_router(sso_auth_router, tags=["SSO Authentication"])  # SSO browser-facing redirects and callbacks

    # -------------------------------------------------------------------------
    # v2 Swagger — filtered to /api/v2/ routes only.
    # -------------------------------------------------------------------------
    @_app.get("/api/v2/openapi.json", include_in_schema=False)
    async def v2_openapi_schema() -> dict:
        from fastapi.openapi.utils import get_openapi as _get_openapi

        v2_routes = [r for r in _app.routes if getattr(r, "path", "").startswith("/api/v2/")]
        return _get_openapi(
            title="TGS API v2",
            version=settings.APP_VERSION,
            routes=v2_routes,
        )

    @_app.get("/api/v2/docs", include_in_schema=False)
    async def v2_swagger_ui():
        from fastapi.openapi.docs import get_swagger_ui_html
        from fastapi.responses import HTMLResponse

        html = get_swagger_ui_html(
            openapi_url="/api/v2/openapi.json",
            title="TGS API v2 — Docs",
        )
        return HTMLResponse(html.body)

    return _app


# Module-level instance used by uvicorn and export_openapi.py.
# create_app() runs at import time but performs no I/O — it only wires routes
# and middleware. All DB/Redis/secret initialisation happens inside lifespan.
app = create_app()
