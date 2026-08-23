from __future__ import annotations

import uuid

from fastapi import APIRouter, Body, Depends, Query, Request, status
from sqlalchemy.orm import Session

from app.api.deps import (
    get_db,
    require_config_or_api_key,
    require_readonly_or_api_key,
)
from app.core.request_auth import ApiKeyPrincipal
from app.models.user import User
from app.schemas.call_flow import (
    FlowDataResponse,
    FlowDataSaveResponse,
    FlowDataUpdate,
    FlowValidationResponse,
    PaginatedFlowDataResponse,
)
from app.services.audit_service import log_audit_event
from app.services.call_flow_service import call_flow_service

router = APIRouter(prefix="/flows", tags=["Visual Flow Editor"])


def _tenant_id(principal: User | ApiKeyPrincipal) -> uuid.UUID:
    return principal.current_tenant_id


@router.get(
    "/flow-data",
    response_model=PaginatedFlowDataResponse,
    status_code=status.HTTP_200_OK,
    summary="List visual and pre-compiled flow graphs across the workspace",
)
def list_flow_data(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100, alias="pageSize"),
    principal: User | ApiKeyPrincipal = Depends(require_readonly_or_api_key),
    db: Session = Depends(get_db),
) -> PaginatedFlowDataResponse:
    return call_flow_service.list_flow_data(
        db, _tenant_id(principal), page, page_size
    )


@router.put(
    "/{flow_id}/flow-data",
    response_model=FlowDataSaveResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate, pre-compile, and save a visual flow graph",
)
def update_flow_data(
    flow_id: uuid.UUID,
    body: FlowDataUpdate,
    request: Request,
    principal: User | ApiKeyPrincipal = Depends(require_config_or_api_key),
    db: Session = Depends(get_db),
) -> FlowDataSaveResponse:
    tenant_id = _tenant_id(principal)
    result = call_flow_service.update_flow_data(db, flow_id, tenant_id, body)

    log_audit_event(
        db,
        request=request,
        tenant_id=tenant_id,
        action="flow_data.updated",
        resource_type="call_flow",
        resource_id=flow_id,
        new_value={"node_count": len(body.flow_data.nodes)},
        actor_user_id=principal.id,
    )
    return result


@router.get(
    "/{flow_id}/flow-data",
    response_model=FlowDataResponse,
    status_code=status.HTTP_200_OK,
    summary="Get the raw and pre-compiled visual flow graph",
)
def get_flow_data(
    flow_id: uuid.UUID,
    mode: str | None = Query(
        None,
        description=(
            "Pass 'readonly' to strip sensitive node fields (prompts, phone numbers, "
            "webhook URLs) and omit the compiled plan from the response. "
            "Any other value (or omitting the parameter) returns the full response."
        ),
    ),
    principal: User | ApiKeyPrincipal = Depends(require_readonly_or_api_key),
    db: Session = Depends(get_db),
) -> FlowDataResponse:
    return call_flow_service.get_flow_data(
        db, flow_id, _tenant_id(principal), readonly=(mode == "readonly")
    )


@router.post(
    "/{flow_id}/validate",
    response_model=FlowValidationResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate the current (or proposed) flow graph without saving",
)
def validate_flow_data(
    flow_id: uuid.UUID,
    body: FlowDataUpdate | None = Body(default=None),
    principal: User | ApiKeyPrincipal = Depends(require_readonly_or_api_key),
    db: Session = Depends(get_db),
) -> FlowValidationResponse:
    return call_flow_service.validate_flow_data(
        db, flow_id, _tenant_id(principal), body
    )


@router.get(
    "/{flow_id}/flow-data/validate",
    response_model=FlowValidationResponse,
    status_code=status.HTTP_200_OK,
    deprecated=True,
    include_in_schema=False,
    summary="[Deprecated] Validate current flow graph",
)
def validate_flow_data_deprecated(
    flow_id: uuid.UUID,
    principal: User | ApiKeyPrincipal = Depends(require_readonly_or_api_key),
    db: Session = Depends(get_db),
) -> FlowValidationResponse:
    """Deprecated alias for GET /{flow_id}/flow-data/validate to support transition to POST /{flow_id}/validate."""
    return call_flow_service.validate_flow_data(
        db, flow_id, _tenant_id(principal), body=None
    )
