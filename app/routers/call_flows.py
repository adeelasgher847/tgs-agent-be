from __future__ import annotations

import uuid
from typing import Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_tenant
from app.core.error_responses import build_api_error_payload
from app.core.request_auth import ApiKeyPrincipal
from app.models.user import User
from app.schemas.call_flow import CallFlowCreate, CallFlowUpdate
from app.services.call_flow_service import call_flow_service

router = APIRouter()


def _workspace_id(principal: Union[User, ApiKeyPrincipal]) -> uuid.UUID:
    return principal.current_tenant_id


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")


def _error_response(
    request: Request,
    code: int,
    message: str,
    *,
    error_code: Optional[str] = None,
) -> JSONResponse:
    payload = build_api_error_payload(
        code,
        message,
        error_code=error_code,
        request_id=_request_id(request),
    )
    return JSONResponse(status_code=code, content=payload)


@router.post("/", status_code=status.HTTP_201_CREATED)
def create_call_flow(
    body: CallFlowCreate,
    principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    result = call_flow_service.create_flow(db, _workspace_id(principal), body)
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=result)


@router.get("/")
def list_call_flows(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100, alias="pageSize"),
    principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    return call_flow_service.list_flows(
        db, _workspace_id(principal), page, page_size
    )


@router.get("/{flow_id}/prompt-versions")
def get_prompt_versions(
    flow_id: uuid.UUID,
    principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    return call_flow_service.get_prompt_versions(db, flow_id, _workspace_id(principal))


@router.get("/{flow_id}")
def get_call_flow(
    flow_id: uuid.UUID,
    principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    return call_flow_service.get_flow(db, flow_id, _workspace_id(principal))


@router.put("/{flow_id}")
def update_call_flow(
    flow_id: uuid.UUID,
    body: CallFlowUpdate,
    principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    return call_flow_service.update_flow(
        db, flow_id, _workspace_id(principal), body
    )


@router.delete("/{flow_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_call_flow(
    flow_id: uuid.UUID,
    request: Request,
    principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    try:
        call_flow_service.delete_flow(db, flow_id, _workspace_id(principal))
    except HTTPException as exc:
        if exc.status_code == status.HTTP_409_CONFLICT:
            return _error_response(
                request,
                status.HTTP_409_CONFLICT,
                str(exc.detail),
                error_code="flow_has_active_calls",
            )
        raise
    return Response(status_code=status.HTTP_204_NO_CONTENT)
