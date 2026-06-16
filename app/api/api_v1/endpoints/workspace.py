"""Workspace (tenant) management endpoints — API-key authenticated.

Every endpoint requires a valid API key resolved by :class:`ApiKeyMiddleware`.
Endpoints addressing a specific workspace return ``403`` if the authenticated
workspace id does not match the URL/target workspace.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_workspace_api_key, require_tenant
from app.core.request_auth import get_workspace_from_request
from app.core.config import settings
from app.core.logger import logger
from app.core.workspace import Workspace
from app.models.tenant import Tenant
from app.repositories.workspace_repository import WorkspaceRepository
from app.schemas.base import SuccessResponse
from app.schemas.integration import MakeSecretResponse, N8nSecretResponse
from app.schemas.workspace import (
    WorkspaceCreate,
    WorkspaceCreatedOut,
    WorkspaceOut,
    WorkspaceUpdateName,
)
from app.services.integration_service import (
    generate_make_secret,
    generate_n8n_secret,
    store_make_secret,
    store_n8n_secret,
)
from app.utils.response import create_success_response

router = APIRouter()

_COMMON_ERROR_RESPONSES: dict = {
    400: {"description": "Validation error — name too short/long or invalid JSON"},
    401: {"description": "Missing or invalid x-api-key / x-workspace-id header"},
    403: {"description": "Workspace mismatch or JWT used on an API-key-only route"},
    404: {"description": "Workspace not found"},
    409: {"description": "Workspace name already taken"},
    429: {"description": "Rate limit exceeded (60 req / 60 s per API key)"},
}


def _repository(db: Session = Depends(get_db)) -> WorkspaceRepository:
    return WorkspaceRepository(db)


_DB_ERROR = HTTPException(
    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    detail="Database error",
)


def _ensure_same_workspace(target: uuid.UUID, authed: uuid.UUID) -> None:
    if target != authed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this workspace",
        )


def _validation_error_detail(exc: ValidationError) -> str:
    errors = exc.errors()
    return errors[0]["msg"] if errors else "Invalid request body"


async def _parse_create_body(request: Request) -> WorkspaceCreate:
    try:
        body = await request.json()
        return WorkspaceCreate.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_validation_error_detail(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON body",
        ) from exc


async def _parse_update_name_body(request: Request) -> WorkspaceUpdateName:
    try:
        body = await request.json()
        return WorkspaceUpdateName.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_validation_error_detail(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON body",
        ) from exc


@router.post(
    "",
    response_model=WorkspaceCreatedOut,
    response_model_by_alias=True,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new workspace",
    responses={**_COMMON_ERROR_RESPONSES, 201: {"description": "Workspace created — flat body: {id, name, createdAt}"}},
)
def create_workspace(
    payload: WorkspaceCreate = Depends(_parse_create_body),
    _: Workspace = Depends(get_workspace_api_key),
    repo: WorkspaceRepository = Depends(_repository),
):
    """Create a new workspace. Name must be unique among active workspaces (3–50 chars)."""
    try:
        if repo.find_by_name(payload.name) is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Workspace with this name already exists",
            )
        tenant = repo.create(payload.name)
    except HTTPException:
        raise
    except IntegrityError as exc:
        logger.warning("Workspace create integrity error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Workspace with this name already exists",
        )
    except SQLAlchemyError as exc:
        logger.error("Workspace create DB error: %s", exc, exc_info=True)
        raise _DB_ERROR

    return WorkspaceCreatedOut.model_validate(tenant)


@router.get(
    "/{workspace_id}",
    response_model=SuccessResponse[WorkspaceOut],
    response_model_by_alias=True,
    summary="Get a workspace by id",
    responses=_COMMON_ERROR_RESPONSES,
)
def get_workspace_by_id(
    workspace_id: uuid.UUID,
    authed: Workspace = Depends(get_workspace_api_key),
    repo: WorkspaceRepository = Depends(_repository),
):
    _ensure_same_workspace(workspace_id, authed.id)

    try:
        tenant = repo.find_by_id(workspace_id)
    except SQLAlchemyError as exc:
        logger.error("Workspace get DB error: %s", exc, exc_info=True)
        raise _DB_ERROR

    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found",
        )

    return create_success_response(
        WorkspaceOut.model_validate(tenant),
        "Workspace retrieved successfully",
    )


@router.put(
    "/name",
    response_model=SuccessResponse[WorkspaceOut],
    response_model_by_alias=True,
    summary="Update the authenticated workspace's name",
    responses=_COMMON_ERROR_RESPONSES,
)
def update_workspace_name(
    payload: WorkspaceUpdateName = Depends(_parse_update_name_body),
    authed: Workspace = Depends(get_workspace_api_key),
    repo: WorkspaceRepository = Depends(_repository),
):
    """Update the name of the authenticated workspace. 409 on duplicate name."""
    try:
        tenant = repo.find_by_id(authed.id)
        if tenant is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workspace not found",
            )

        if tenant.name != payload.name:
            existing = repo.find_by_name(payload.name)
            if existing is not None and existing.id != tenant.id:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Workspace with this name already exists",
                )
            tenant = repo.update_name(tenant, payload.name)
    except HTTPException:
        raise
    except IntegrityError as exc:
        logger.warning("Workspace update integrity error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Workspace with this name already exists",
        )
    except SQLAlchemyError as exc:
        logger.error("Workspace update DB error: %s", exc, exc_info=True)
        raise _DB_ERROR

    return create_success_response(
        WorkspaceOut.model_validate(tenant),
        "Workspace name updated successfully",
    )


@router.post(
    "/settings/make-secret",
    response_model=MakeSecretResponse,
    summary="Generate (or rotate) the Make.com integration secret for this workspace",
    description=(
        "Generates a new 64-character hex secret and stores it in workspace_settings. "
        "Calling this again rotates the secret — old scenarios must be updated. "
        "Returns the new secret and the webhook URL to configure in Make.com."
    ),
    responses={
        200: {
            "description": "Secret generated",
            "content": {
                "application/json": {
                    "example": {
                        "secret": "a3f2...hex64...",
                        "webhook_url": "https://example.com/api/v1/integrations/make/trigger",
                    }
                }
            },
        },
        **_COMMON_ERROR_RESPONSES,
    },
)
def generate_make_integration_secret(
    request: Request,
    _user=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Generate or rotate the Make.com webhook secret for the authenticated workspace."""
    workspace = get_workspace_from_request(request)
    if workspace is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Workspace context not available")

    tenant = db.query(Tenant).filter(Tenant.id == workspace.id).first()
    if tenant is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")

    secret = generate_make_secret()
    store_make_secret(db, tenant, secret)

    base_url = settings.WEBHOOK_BASE_URL.rstrip("/")
    return MakeSecretResponse(
        secret=secret,
        webhook_url=f"{base_url}/api/v1/integrations/make/trigger",
    )


@router.post(
    "/settings/n8n-secret",
    response_model=N8nSecretResponse,
    summary="Generate (or rotate) the n8n integration secret for this workspace",
    description=(
        "Generates a new 64-character hex secret and stores it in workspace_settings. "
        "Calling this again rotates the secret — existing n8n workflows must be updated. "
        "Returns the new secret and the webhook URL to configure in n8n."
    ),
    responses={
        200: {
            "description": "Secret generated",
            "content": {
                "application/json": {
                    "example": {
                        "secret": "a3f2...hex64...",
                        "webhook_url": "https://example.com/api/v1/integrations/n8n/trigger",
                    }
                }
            },
        },
        **_COMMON_ERROR_RESPONSES,
    },
)
def generate_n8n_integration_secret(
    request: Request,
    _user=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Generate or rotate the n8n webhook secret for the authenticated workspace."""
    workspace = get_workspace_from_request(request)
    if workspace is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Workspace context not available")

    tenant = db.query(Tenant).filter(Tenant.id == workspace.id).first()
    if tenant is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")

    secret = generate_n8n_secret()
    store_n8n_secret(db, tenant, secret)

    base_url = settings.WEBHOOK_BASE_URL.rstrip("/")
    return N8nSecretResponse(
        secret=secret,
        webhook_url=f"{base_url}/api/v1/integrations/n8n/trigger",
    )


@router.delete(
    "/{workspace_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Soft delete a workspace",
    responses={**_COMMON_ERROR_RESPONSES, 204: {"description": "Workspace soft-deleted, no body"}},
)
def soft_delete_workspace(
    workspace_id: uuid.UUID,
    authed: Workspace = Depends(get_workspace_api_key),
    repo: WorkspaceRepository = Depends(_repository),
):
    _ensure_same_workspace(workspace_id, authed.id)

    try:
        tenant = repo.find_by_id(workspace_id)
        if tenant is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workspace not found",
            )
        repo.soft_delete(tenant)
    except HTTPException:
        raise
    except SQLAlchemyError as exc:
        logger.error("Workspace delete DB error: %s", exc, exc_info=True)
        raise _DB_ERROR

    return Response(status_code=status.HTTP_204_NO_CONTENT)
