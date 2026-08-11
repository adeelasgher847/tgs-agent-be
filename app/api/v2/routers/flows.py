"""
v2 A/B prompt testing + cross-session caller memory endpoints.

Auth: any authenticated tenant principal (API key or JWT); config-rank required
to mutate A/B settings, read-only rank is sufficient to view results. Caller
memory settings require admin rank (owner-equivalent — see require_admin_or_api_key).

PUT  /api/v2/flows/{flow_id}/ab-test
GET  /api/v2/flows/{flow_id}/ab-results
PUT  /api/v2/flows/{flow_id}/ab-test/winner
PUT  /api/v2/flows/{flow_id}/caller-memory-settings
PUT  /api/v2/flows/{flow_id}/post-call-actions-settings

Visual Flow Editor endpoints (flow-data, flow-data/validate) live in
app.api.v2.routers.flow_data — a separate router under the same prefix.

Note: the caller memory settings path is deliberately NOT `/{flow_id}/settings` —
that path is already registered by app.api.v2.routers.hipaa for the HIPAA
compliance toggle, and reusing it here would silently shadow that endpoint.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.orm import Session

from app.api.deps import (
    get_db,
    require_admin_or_api_key,
    require_config_or_api_key,
    require_readonly_or_api_key,
)
from app.core.request_auth import ApiKeyPrincipal
from app.models.user import User
from app.schemas.ab_testing import (
    AbResultsResponse,
    AbTestResponse,
    AbTestUpdate,
    AbTestWinnerUpdate,
)
from app.schemas.call_flow import (
    CallerMemorySettingsResponse,
    CallerMemorySettingsUpdate,
    PostCallActionsSettingsResponse,
    PostCallActionsSettingsUpdate,
)
from app.services.audit_service import log_audit_event
from app.services.call_flow_service import call_flow_service

router = APIRouter(prefix="/flows", tags=["A/B Prompt Testing"])


def _tenant_id(principal: User | ApiKeyPrincipal) -> uuid.UUID:
    return principal.current_tenant_id


@router.put(
    "/{flow_id}/ab-test",
    response_model=AbTestResponse,
    status_code=status.HTTP_200_OK,
    summary="Configure A/B prompt testing on a call flow",
)
def update_ab_test(
    flow_id: uuid.UUID,
    body: AbTestUpdate,
    principal: User | ApiKeyPrincipal = Depends(require_config_or_api_key),
    db: Session = Depends(get_db),
) -> AbTestResponse:
    return call_flow_service.update_ab_test(db, flow_id, _tenant_id(principal), body)


@router.get(
    "/{flow_id}/ab-results",
    response_model=AbResultsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get A/B prompt test results and statistical significance",
)
def get_ab_results(
    flow_id: uuid.UUID,
    principal: User | ApiKeyPrincipal = Depends(require_readonly_or_api_key),
    db: Session = Depends(get_db),
) -> AbResultsResponse:
    return call_flow_service.get_ab_results(db, flow_id, _tenant_id(principal))


@router.put(
    "/{flow_id}/ab-test/winner",
    status_code=status.HTTP_200_OK,
    summary="Promote the winning A/B variant to the flow's active prompt",
)
def promote_ab_winner(
    flow_id: uuid.UUID,
    body: AbTestWinnerUpdate,
    principal: User | ApiKeyPrincipal = Depends(require_config_or_api_key),
    db: Session = Depends(get_db),
) -> dict:
    return call_flow_service.promote_ab_winner(db, flow_id, _tenant_id(principal), body)


@router.put(
    "/{flow_id}/caller-memory-settings",
    response_model=CallerMemorySettingsResponse,
    status_code=status.HTTP_200_OK,
    summary="Configure cross-session caller memory on a call flow",
)
def update_caller_memory_settings(
    flow_id: uuid.UUID,
    body: CallerMemorySettingsUpdate,
    request: Request,
    principal: User | ApiKeyPrincipal = Depends(require_admin_or_api_key),
    db: Session = Depends(get_db),
) -> CallerMemorySettingsResponse:
    tenant_id = _tenant_id(principal)
    result = call_flow_service.update_caller_memory_settings(db, flow_id, tenant_id, body)

    log_audit_event(
        db,
        request=request,
        tenant_id=tenant_id,
        action="caller_memory_settings.updated",
        resource_type="call_flow",
        resource_id=flow_id,
        new_value={
            "caller_memory_enabled": result.caller_memory_enabled,
            "caller_memory_window": result.caller_memory_window,
        },
        actor_user_id=principal.id,
    )
    return result


@router.put(
    "/{flow_id}/post-call-actions-settings",
    response_model=PostCallActionsSettingsResponse,
    status_code=status.HTTP_200_OK,
    summary="Configure post-call email summary actions on a call flow",
    description=(
        "Configures the two \"Post Call Actions\" toggles for this call flow:\n\n"
        "- **Email Summary** (`email_summary_enabled` + `email_summary_recipients`): "
        "sends the call summary and extracted variables to an explicit list of "
        "recipient emails after each completed call.\n"
        "- **Summary to Business Owner** (`summary_to_business_owner_enabled`): sends "
        "the same email to the tenant's business owner (the workspace creator's "
        "account email) — there is no separate address field for this, the recipient "
        "is resolved server-side.\n\n"
        "Both toggles are independent and may be enabled together; recipients from "
        "both sources are deduplicated before sending. The request body always fully "
        "replaces all three fields (send the current values for any field you don't "
        "want to change). Sending is fire-and-forget after call completion — this "
        "endpoint only updates the stored configuration; it does not send an email "
        "itself. Requires admin rank."
    ),
)
def update_post_call_actions_settings(
    flow_id: uuid.UUID,
    body: PostCallActionsSettingsUpdate,
    request: Request,
    principal: User | ApiKeyPrincipal = Depends(require_admin_or_api_key),
    db: Session = Depends(get_db),
) -> PostCallActionsSettingsResponse:
    tenant_id = _tenant_id(principal)
    result = call_flow_service.update_post_call_actions_settings(db, flow_id, tenant_id, body)

    log_audit_event(
        db,
        request=request,
        tenant_id=tenant_id,
        action="post_call_actions_settings.updated",
        resource_type="call_flow",
        resource_id=flow_id,
        new_value={
            "email_summary_enabled": result.email_summary_enabled,
            "email_summary_recipients": result.email_summary_recipients,
            "summary_to_business_owner_enabled": result.summary_to_business_owner_enabled,
        },
        actor_user_id=principal.id,
    )
    return result
