from __future__ import annotations

import uuid
from typing import Optional, Union

from fastapi import APIRouter, Body, Depends, Request, status
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
    FlowDataUpdate,
    FlowValidationResponse,
)
from app.services.audit_service import log_audit_event
from app.services.call_flow_service import call_flow_service

router = APIRouter(prefix="/flows", tags=["Visual Flow Editor"])


def _tenant_id(principal: Union[User, ApiKeyPrincipal]) -> uuid.UUID:
    return principal.current_tenant_id


@router.put(
    "/{flow_id}/flow-data",
    response_model=FlowDataResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate, pre-compile, and save a visual flow graph",
)
def update_flow_data(
    flow_id: uuid.UUID,
    body: FlowDataUpdate,
    request: Request,
    principal: Union[User, ApiKeyPrincipal] = Depends(require_config_or_api_key),
    db: Session = Depends(get_db),
) -> FlowDataResponse:
    tenant_id = _tenant_id(principal)
    result = call_flow_service.update_flow_data(db, flow_id, tenant_id, body)

    log_audit_event(
        db,
        request=request,
        tenant_id=tenant_id,
        action="flow_data.updated",
        resource_type="call_flow",
        resource_id=flow_id,
        new_value={
            "node_count": (
                len(result.flow_data.get("nodes", [])) if result.flow_data else 0
            )
        },
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
    principal: Union[User, ApiKeyPrincipal] = Depends(require_readonly_or_api_key),
    db: Session = Depends(get_db),
) -> FlowDataResponse:
    return call_flow_service.get_flow_data(db, flow_id, _tenant_id(principal))


@router.get(
    "/{flow_id}/flow-data/validate",
    response_model=FlowValidationResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate the current (or posted) flow graph without saving",
)
def validate_flow_data(
    flow_id: uuid.UUID,
    body: Optional[FlowDataUpdate] = Body(default=None),
    principal: Union[User, ApiKeyPrincipal] = Depends(require_readonly_or_api_key),
    db: Session = Depends(get_db),
) -> FlowValidationResponse:
    return call_flow_service.validate_flow_data(
        db, flow_id, _tenant_id(principal), body
    )
