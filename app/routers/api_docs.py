"""API documentation — committed OpenAPI spec + Swagger UI at GET /api/docs."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse

from app.api.docs_auth import verify_api_docs_basic

router = APIRouter()

# Repo root: app/routers/api_docs.py → parents[2]
_OPENAPI_YAML = (
    Path(__file__).resolve().parents[2] / "docs" / "api" / "openapi.yaml"
)


@router.get("/docs", include_in_schema=False)
async def api_docs(_: None = Depends(verify_api_docs_basic)) -> HTMLResponse:
    """Swagger UI for the API contract (spec at /api/docs/openapi.yaml)."""
    return get_swagger_ui_html(
        openapi_url="/api/docs/openapi.yaml",
        title="TGS Voice Agent API",
    )


@router.get("/docs/openapi.yaml", include_in_schema=False)
async def api_docs_openapi_yaml(
    _: None = Depends(verify_api_docs_basic),
) -> FileResponse:
    """Serve the committed OpenAPI 3.0 YAML contract."""
    return FileResponse(
        _OPENAPI_YAML,
        media_type="application/yaml",
        filename="openapi.yaml",
    )
