from __future__ import annotations

import uuid
from typing import Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin_or_owner, require_tenant
from app.core.error_responses import build_api_error_payload
from app.core.request_auth import ApiKeyPrincipal
from app.models.call_flow import CallFlow
from app.models.knowledge_base_document import KnowledgeBase
from app.models.user import User
from app.schemas.call_flow import CallFlowCreate, CallFlowSettingsUpdate, CallFlowUpdate
from app.schemas.knowledge_base import FlowKbUpdate
from app.services.audit_service import log_audit_event
from app.services.call_flow_service import call_flow_service
from app.utils.response import create_success_response

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
    request: Request,
    principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    tid = _workspace_id(principal)
    result = call_flow_service.create_flow(db, tid, body)
    log_audit_event(
        db,
        request=request,
        tenant_id=tid,
        action="call_flow.created",
        resource_type="call_flow",
        resource_id=result.get("id") if isinstance(result, dict) else None,
        new_value=body.model_dump(exclude_none=True),
        actor_user_id=principal.id if isinstance(principal, User) else None,
    )
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
    request: Request,
    principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    tid = _workspace_id(principal)
    old_flow = call_flow_service.get_flow(db, flow_id, tid)
    result = call_flow_service.update_flow(db, flow_id, tid, body)
    log_audit_event(
        db,
        request=request,
        tenant_id=tid,
        action="call_flow.updated",
        resource_type="call_flow",
        resource_id=flow_id,
        old_value=old_flow if isinstance(old_flow, dict) else None,
        new_value=body.model_dump(exclude_none=True),
        actor_user_id=principal.id if isinstance(principal, User) else None,
    )
    return result


@router.put("/{flow_id}/settings")
def update_call_flow_settings(
    flow_id: uuid.UUID,
    body: CallFlowSettingsUpdate,
    request: Request,
    principal: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    """Toggle Web SDK public access for a flow. Requires admin or owner role."""
    tid = _workspace_id(principal)
    old_flow = call_flow_service.get_flow(db, flow_id, tid)
    result = call_flow_service.update_settings(db, flow_id, tid, body)
    log_audit_event(
        db,
        request=request,
        tenant_id=tid,
        action="call_flow.settings_updated",
        resource_type="call_flow",
        resource_id=flow_id,
        old_value={"public_access": old_flow.get("publicAccess")} if isinstance(old_flow, dict) else None,
        new_value={"public_access": body.public_access},
        actor_user_id=principal.id if isinstance(principal, User) else None,
    )
    return result


@router.delete("/{flow_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_call_flow(
    flow_id: uuid.UUID,
    request: Request,
    principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    tid = _workspace_id(principal)
    old_flow = call_flow_service.get_flow(db, flow_id, tid)
    try:
        call_flow_service.delete_flow(db, flow_id, tid)
    except HTTPException as exc:
        if exc.status_code == status.HTTP_409_CONFLICT:
            return _error_response(
                request,
                status.HTTP_409_CONFLICT,
                str(exc.detail),
                error_code="flow_has_active_calls",
            )
        raise
    log_audit_event(
        db,
        request=request,
        tenant_id=tid,
        action="call_flow.deleted",
        resource_type="call_flow",
        resource_id=flow_id,
        old_value=old_flow if isinstance(old_flow, dict) else None,
        actor_user_id=principal.id if isinstance(principal, User) else None,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.put("/{flow_id}/knowledge-bases")
def update_flow_knowledge_bases(
    flow_id: uuid.UUID,
    body: FlowKbUpdate,
    principal: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    """Replace the list of KB IDs attached to a call flow.

    Each supplied kb_id is validated to belong to the same workspace.
    Passing an empty list detaches all KBs.
    """
    workspace_id = _workspace_id(principal)

    flow = (
        db.query(CallFlow)
        .filter(
            CallFlow.id == flow_id,
            CallFlow.tenant_id == workspace_id,
            CallFlow.is_deleted == False,  # noqa: E712
        )
        .first()
    )
    if flow is None:
        raise HTTPException(status_code=404, detail=f"Call flow {flow_id} not found")

    # Validate every supplied KB belongs to this workspace
    for kb_id in body.kb_ids:
        kb = (
            db.query(KnowledgeBase)
            .filter(
                KnowledgeBase.id == kb_id,
                KnowledgeBase.workspace_id == workspace_id,
            )
            .first()
        )
        if kb is None:
            raise HTTPException(
                status_code=404,
                detail=f"Knowledge base {kb_id} not found in this workspace",
            )

    flow.knowledge_base_ids = [str(k) for k in body.kb_ids]
    db.commit()
    db.refresh(flow)

    return create_success_response(
        {"flow_id": str(flow_id), "kb_ids": flow.knowledge_base_ids},
        "Knowledge bases updated",
    )
