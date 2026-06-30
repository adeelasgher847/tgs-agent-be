"""
Integration trigger endpoints for Make.com and n8n.

Both endpoints share the same internal call dispatch (voice_call_service.initiate_call).
Auth is handled per-integration:
  - Make.com: X-Make-Secret header validated against workspace_settings.make_secret
  - n8n:      X-N8N-Webhook-Secret header validated against workspace_settings.n8n_secret

Rate limit: 10 integration-triggered calls per minute per workspace.
"""
from __future__ import annotations

import hmac
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_tenant
from app.core.config import settings
from app.core.logger import logger
from app.schemas.integration import (
    IntegrationItem,
    IntegrationListResponse,
    MakeTriggerRequest,
    MakeTriggerResponse,
    N8nTriggerResponse,
)
from app.schemas.twilio import CallInitiateRequest
from app.services.integration_service import (
    check_integration_rate_limit,
    get_last_triggered_at,
    get_make_secret,
    get_n8n_secret,
    record_last_triggered,
    resolve_tenant_by_agent,
)
from app.services.voice_call_service import initiate_call as initiate_call_service

router = APIRouter()


def _build_internal_request(webhook_secret: str) -> Request:
    """
    Build a minimal Starlette Request for internal outbound call dispatch.

    Mirrors the pattern in batch_call_worker_service._build_fake_request so that
    verify_n8n_webhook_secret_async resolves to True and initiate_call uses the
    tenant_id from the body rather than requiring a JWT user.
    """
    from starlette.types import Scope

    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/internal/integration",
        "query_string": b"",
        "headers": [
            (b"x-n8n-webhook-secret", webhook_secret.encode("latin-1")),
            (b"content-type", b"application/json"),
        ],
        "state": {},
    }

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive=_receive)


def _rate_limit_response(retry_after: float) -> JSONResponse:
    retry_dt = datetime.fromtimestamp(retry_after, tz=timezone.utc)
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": {
                "code": "rate_limit_exceeded",
                "message": "Integration call limit exceeded. Maximum 10 calls per minute per workspace.",
                "retry_after": retry_dt.isoformat().replace("+00:00", "Z"),
            }
        },
    )


@router.post(
    "/make/trigger",
    response_model=MakeTriggerResponse,
    summary="Make.com — trigger an outbound call",
    description=(
        "Trigger an outbound call from a Make.com scenario. "
        "Supply the workspace-specific secret in the `X-Make-Secret` header. "
        "Rate limited to 10 calls per minute per workspace."
    ),
    responses={
        200: {
            "description": "Call initiated",
            "content": {"application/json": {"example": {"call_id": "uuid", "status": "initiated"}}},
        },
        403: {
            "description": "Invalid secret",
            "content": {
                "application/json": {
                    "example": {"message": "Invalid secret", "code": "unauthorized"}
                }
            },
        },
        429: {"description": "Rate limit exceeded"},
    },
    tags=["Integrations"],
)
async def make_trigger(
    body: MakeTriggerRequest,
    request: Request,
    db: Session = Depends(get_db),
    x_make_secret: Optional[str] = Header(default=None, alias="X-Make-Secret"),
) -> MakeTriggerResponse:
    # 1. Resolve agent and tenant from agent_id in body
    agent, tenant = resolve_tenant_by_agent(db, body.agent_id)
    if agent is None or tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )

    # 2. Validate X-Make-Secret against workspace_settings
    stored_secret = get_make_secret(tenant)
    if not stored_secret or not x_make_secret or not hmac.compare_digest(x_make_secret, stored_secret):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"message": "Invalid secret", "code": "unauthorized"},
        )

    # 3. Per-workspace rate limit
    allowed, retry_after = await check_integration_rate_limit(tenant.id)
    if not allowed:
        return _rate_limit_response(retry_after)

    # 4. Build CallInitiateRequest — map Make body fields to internal schema
    call_request = CallInitiateRequest(
        agentId=body.agent_id,
        toNumber=body.to_number,
        tenant_id=str(tenant.id),
        jd_context=body.variables,
    )

    # 5. Dispatch via shared call service using the established internal dispatch
    #    pattern (same as batch worker and callback scheduler).
    #    N8N_WEBHOOK_SECRET must be configured — it is the system's internal dispatch key.
    if not settings.N8N_WEBHOOK_SECRET:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Integration dispatch is not configured on this server. Contact your administrator.",
        )

    fake_request = _build_internal_request(settings.N8N_WEBHOOK_SECRET)
    result = await initiate_call_service(call_request, fake_request, None, db)

    # 6. Record last triggered timestamp (best-effort)
    try:
        record_last_triggered(db, tenant, "make")
    except Exception as exc:
        logger.warning("Failed to record make last_triggered_at: %s", exc)

    # 7. Extract call_id from the SuccessResponse envelope
    if isinstance(result, JSONResponse):
        return result  # propagate errors from initiate_call

    call_data = result.data
    return MakeTriggerResponse(
        call_id=call_data.callSessionId,
        status=call_data.status,
    )


@router.post(
    "/n8n/trigger",
    response_model=N8nTriggerResponse,
    summary="n8n — trigger an outbound call",
    description=(
        "Trigger an outbound call from an n8n HTTP Request node. "
        "Body is identical to `POST /api/v1/voice/call/initiate`. "
        "Requires `X-N8N-Webhook-Secret` header containing the workspace-specific secret. "
        "Rate limited to 10 calls per minute per workspace."
    ),
    responses={
        200: {
            "description": "Call initiated",
            "content": {
                "application/json": {
                    "example": {"success": True, "data": {"call_id": "uuid", "status": "initiated"}}
                }
            },
        },
        403: {"description": "Missing or invalid webhook secret"},
        429: {"description": "Rate limit exceeded"},
    },
    tags=["Integrations"],
)
async def n8n_trigger(
    body: CallInitiateRequest,
    request: Request,
    db: Session = Depends(get_db),
    x_n8n_webhook_secret: Optional[str] = Header(default=None, alias="X-N8N-Webhook-Secret"),
) -> N8nTriggerResponse:
    # 1. Resolve agent and owning tenant from agentId — same pattern as Make.com.
    #    This ensures tenant resolution is authoritative (not caller-supplied tenant_id).
    agent, tenant = resolve_tenant_by_agent(db, body.agentId)
    if agent is None or tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )

    # 2. Validate X-N8N-Webhook-Secret against the per-workspace n8n secret.
    #    The global N8N_WEBHOOK_SECRET is reserved for internal system dispatch only
    #    and must never authorize public external calls.
    stored_secret = get_n8n_secret(tenant)
    if not stored_secret or not x_n8n_webhook_secret or not hmac.compare_digest(x_n8n_webhook_secret, stored_secret):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"message": "Invalid or missing webhook secret", "code": "unauthorized"},
        )

    # 3. Bind tenant_id from the authoritative agent lookup so downstream dispatch
    #    cannot be redirected to a different tenant by a manipulated body field.
    body.tenant_id = str(tenant.id)

    # 4. Per-workspace rate limit
    allowed, retry_after = await check_integration_rate_limit(tenant.id)
    if not allowed:
        return _rate_limit_response(retry_after)

    # 5. Dispatch via shared call service using internal fake request.
    if not settings.N8N_WEBHOOK_SECRET:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Integration dispatch is not configured on this server. Contact your administrator.",
        )

    fake_request = _build_internal_request(settings.N8N_WEBHOOK_SECRET)
    result = await initiate_call_service(body, fake_request, None, db)

    # 6. Record last triggered (best-effort)
    try:
        record_last_triggered(db, tenant, "n8n")
    except Exception as exc:
        logger.warning("Failed to record n8n last_triggered_at: %s", exc)

    # 7. Wrap in n8n-style envelope
    if isinstance(result, JSONResponse):
        return result  # propagate errors

    call_data = result.data
    return N8nTriggerResponse(
        success=True,
        data={
            "call_id": call_data.callSessionId,
            "status": call_data.status,
        },
    )


@router.get(
    "",
    response_model=IntegrationListResponse,
    summary="List available integrations and their connection status",
    description=(
        "Returns Make.com and n8n integration status for the authenticated workspace. "
        "Connected = secret is configured. webhook_url is the endpoint to configure in each tool."
    ),
    tags=["Integrations"],
)
async def list_integrations(
    request: Request,
    user=Depends(require_tenant),
    db: Session = Depends(get_db),
) -> IntegrationListResponse:
    # Resolve tenant from authenticated user/API key
    from app.core.request_auth import get_workspace_from_request
    from app.models.tenant import Tenant as TenantModel

    workspace = get_workspace_from_request(request)
    if workspace is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Workspace context not available",
        )

    tenant = db.query(TenantModel).filter(TenantModel.id == workspace.id).first()
    if tenant is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")

    base_url = settings.WEBHOOK_BASE_URL.rstrip("/")

    make_secret = get_make_secret(tenant)
    n8n_secret = get_n8n_secret(tenant)

    from app.services import hubspot_service

    hubspot_connected, hubspot_connected_at = hubspot_service.get_connection_status(db, tenant.id)

    integrations = [
        IntegrationItem(
            name="make",
            connected=bool(make_secret),
            webhook_url=f"{base_url}/api/v1/integrations/make/trigger",
            last_triggered_at=get_last_triggered_at(tenant, "make"),
        ),
        IntegrationItem(
            name="n8n",
            connected=bool(n8n_secret),
            webhook_url=f"{base_url}/api/v1/integrations/n8n/trigger",
            last_triggered_at=get_last_triggered_at(tenant, "n8n"),
        ),
        IntegrationItem(
            name="hubspot",
            connected=hubspot_connected,
            connected_at=hubspot_connected_at,
        ),
    ]

    return IntegrationListResponse(integrations=integrations)
