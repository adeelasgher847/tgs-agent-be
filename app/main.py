from fastapi import FastAPI, Request, HTTPException, Query, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from typing import Optional
from twilio.twiml.voice_response import VoiceResponse
from sqlalchemy.orm import Session
from fastapi.openapi.utils import get_openapi

from app.api.api_v1.api import api_router
from app.routers.health import router as health_router
# Removed old call_session_websocket import - now using general_websocket via api_router
from app.schemas.base import SuccessResponse
from app.utils.response import create_success_response
from app.utils.rate_limiter import init_rate_limiter, close_rate_limiter

from app.core.logger import setup_logging, logger
from app.core.pii_redactor import redact_pii
from app.middleware.api_key_middleware import ApiKeyMiddleware

# Initialize centralized logging
setup_logging()

# Use OpenAPI 3.0.x for better Swagger multipart file rendering.
app = FastAPI(openapi_version="3.0.3")


def _patch_binary_formats(schema_obj):
    """
    FastAPI/Pydantic can emit contentMediaType for file fields, which Swagger UI
    sometimes renders as array<string>. Convert these to format=binary for
    consistent file upload widgets.
    """
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
# Global exception handlers — sanitise all error detail before it leaves the
# process so raw exception messages (which may contain PII) are never returned.
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    safe_detail = redact_pii(exc.detail) if isinstance(exc.detail, (str, dict, list)) else exc.detail
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": safe_detail},
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    safe_errors = redact_pii(exc.errors())
    return JSONResponse(status_code=422, content={"detail": safe_errors})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception on %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again later."},
    )


# Initialize rate limiter on startup (temporarily disabled due to Redis connection issues)
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup initiated")
    try:
        await init_rate_limiter()
        logger.info("Rate limiter initialized successfully")
    except Exception as e:
        logger.warning(f"Rate limiter initialization failed: {e}")
        logger.warning("Continuing without rate limiting...")

@app.on_event("shutdown")
async def shutdown_event():
    try:
        await close_rate_limiter()
        logger.info("Rate limiter closed successfully")
    except Exception as e:
        logger.error(f"Rate limiter cleanup failed: {e}")

# API key auth middleware — must be added before CORS so it runs after CORS in the stack
app.add_middleware(ApiKeyMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Your frontend dev server
        "http://localhost:3000",  # Alternative frontend port
        "http://127.0.0.1:5173",  # Alternative localhost
        "http://127.0.0.1:3000",  # Alternative localhost
        "http://192.168.0.121:5173",  # Your IP with frontend port
        "http://192.168.15.129:5173",
        "*"  # Allow all origins (for development only)
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/", response_model=SuccessResponse[dict])
def read_root():
    return create_success_response(
        {"message": "Welcome to the Multi-Tenant SaaS Voice Agent Backend!"},
        "API is running successfully"
    )   
    
app.include_router(api_router, prefix="/api/v1")
app.include_router(health_router)