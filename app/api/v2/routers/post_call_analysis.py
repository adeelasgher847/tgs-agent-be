"""
v2 Post-Call Analysis endpoints — tenant-defined custom variable extraction.

Auth: admin rank required to mutate settings (require_admin_or_api_key), same
rank as the neighboring caller-memory and post-call-actions settings in
app.api.v2.routers.flows.

PUT  /api/v2/flows/{flow_id}/post-call-analysis-settings

This is a separate router file (rather than folded into flows.py) per the
"New feature gets its own router file" convention documented alongside the
other flows-prefixed router files.
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin_or_api_key
from app.core.error_responses import build_api_error_payload
from app.core.request_auth import ApiKeyPrincipal
from app.models.user import User
from app.schemas.call_flow import (
    PostCallAnalysisSettingsResponse,
    PostCallAnalysisSettingsUpdate,
)
from app.services.agent_service import agent_service
from app.services.audit_service import log_audit_event
from app.services.call_flow_service import call_flow_service

router = APIRouter(prefix="/flows", tags=["Post-Call Analysis"])


def _tenant_id(principal: User | ApiKeyPrincipal) -> uuid.UUID:
    return principal.current_tenant_id


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")


def _error_response(
    request: Request,
    status_code: int,
    message: str,
    *,
    error_code: str | None = None,
    extras: dict[str, Any] | None = None,
) -> JSONResponse:
    payload = build_api_error_payload(
        status_code,
        message,
        error_code=error_code,
        request_id=_request_id(request),
        extras=extras,
    )
    return JSONResponse(status_code=status_code, content=payload)


@router.put(
    "/{flow_id}/post-call-analysis-settings",
    response_model=PostCallAnalysisSettingsResponse,
    status_code=status.HTTP_200_OK,
    summary="Configure tenant-defined custom variable extraction on a call flow",
    description=(
        "Configures the \"Post-Call Analysis\" section: a list of custom "
        "variables (name + description) extracted from each completed call by "
        "a dedicated LLM pass, plus the model used for extraction.\n\n"
        "There is no explicit enable toggle — an empty `variablesToExtract` "
        "list disables this feature entirely and leaves the existing "
        "automatic call-summary generation untouched. `analysisModel`, when "
        "provided, must be an active model-catalog entry; when omitted the "
        "flow falls back to the agent's preferred model and then a built-in "
        "chain at extraction time. Requires admin rank."
    ),
)
def update_post_call_analysis_settings(
    flow_id: uuid.UUID,
    body: PostCallAnalysisSettingsUpdate,
    request: Request,
    principal: User | ApiKeyPrincipal = Depends(require_admin_or_api_key),
    db: Session = Depends(get_db),
) -> PostCallAnalysisSettingsResponse:
    tenant_id = _tenant_id(principal)
    try:
        result = call_flow_service.update_post_call_analysis_settings(
            db, flow_id, tenant_id, body
        )
    except HTTPException as exc:
        if exc.status_code == status.HTTP_400_BAD_REQUEST and "LLM model" in str(exc.detail):
            return _error_response(
                request,
                status.HTTP_400_BAD_REQUEST,
                str(exc.detail),
                error_code="invalid_llm_model",
                extras={"allowedValues": agent_service.list_active_llm_model_names(db)},
            )
        raise

    log_audit_event(
        db,
        request=request,
        tenant_id=tenant_id,
        action="post_call_analysis_settings.updated",
        resource_type="call_flow",
        resource_id=flow_id,
        new_value={
            "variables_to_extract": [
                v.model_dump() for v in result.variables_to_extract
            ],
            "analysis_model": result.analysis_model,
        },
        actor_user_id=principal.id,
    )
    return result
