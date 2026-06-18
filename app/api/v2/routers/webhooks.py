"""
v2 Webhooks router.

Auth: any authenticated tenant principal — API key or JWT with non-readonly role
for write operations.

POST   /webhooks                          — create endpoint
GET    /webhooks                          — list endpoints (no secret returned)
DELETE /webhooks/{endpoint_id}           — remove endpoint
POST   /webhooks/{endpoint_id}/test      — send test ping
GET    /webhooks/{endpoint_id}/deliveries — paginated delivery log
"""
from __future__ import annotations

import uuid
from typing import Union

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_workspace, require_tenant
from app.core.request_auth import ApiKeyPrincipal
from app.core.workspace import Workspace
from app.models.user import User
from app.services.audit_service import log_audit_event
from app.schemas.webhook import (
    PaginatedWebhookDeliveries,
    WebhookDeliveryOut,
    WebhookEndpointCreate,
    WebhookEndpointOut,
)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


# ── Service factories ─────────────────────────────────────────────────────────

def _webhook_service(
    workspace: Workspace = Depends(get_workspace),
    _principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Read + write access for any authenticated principal."""
    from app.services.webhook_service import WebhookService

    yield WebhookService(db)


def _webhook_service_write(
    request: Request,
    workspace: Workspace = Depends(get_workspace),
    principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Write access — blocks readonly JWT users on mutating methods."""
    from app.services.role_service import get_user_role_in_tenant
    from app.services.webhook_service import WebhookService

    if isinstance(principal, User) and principal.current_tenant_id:
        role = get_user_role_in_tenant(db, principal.id, principal.current_tenant_id)
        if role and role.name == "readonly":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Read-only access cannot modify resources",
            )

    yield WebhookService(db)


# ── POST /webhooks ────────────────────────────────────────────────────────────

@router.post("", response_model=WebhookEndpointOut, status_code=status.HTTP_201_CREATED)
def create_webhook_endpoint(
    body: WebhookEndpointCreate,
    request: Request,
    workspace: Workspace = Depends(get_workspace),
    db: Session = Depends(get_db),
    svc=Depends(_webhook_service_write),
) -> WebhookEndpointOut:
    """
    Register a new webhook endpoint for the workspace.

    The secret is stored encrypted and never returned. Minimum 16 characters.
    URL must use HTTPS.
    """
    endpoint = svc.create_endpoint(
        workspace_id=workspace.id,
        url=str(body.url),
        raw_secret=body.secret,
    )
    log_audit_event(
        db,
        request=request,
        tenant_id=workspace.id,
        action="webhook_endpoint.created",
        resource_type="webhook_endpoint",
        resource_id=endpoint.id,
        new_value={"url": str(body.url)},
    )
    return WebhookEndpointOut.model_validate(endpoint)


# ── GET /webhooks ─────────────────────────────────────────────────────────────

@router.get("", response_model=list[WebhookEndpointOut])
def list_webhook_endpoints(
    workspace: Workspace = Depends(get_workspace),
    svc=Depends(_webhook_service),
) -> list[WebhookEndpointOut]:
    """List all webhook endpoints for the workspace. Secret is never returned."""
    endpoints = svc.list_endpoints(workspace.id)
    return [WebhookEndpointOut.model_validate(e) for e in endpoints]


# ── DELETE /webhooks/{endpoint_id} ───────────────────────────────────────────

@router.delete("/{endpoint_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_webhook_endpoint(
    endpoint_id: uuid.UUID,
    request: Request,
    workspace: Workspace = Depends(get_workspace),
    db: Session = Depends(get_db),
    svc=Depends(_webhook_service_write),
) -> None:
    """Remove a webhook endpoint and all its delivery logs."""
    svc.delete_endpoint(workspace.id, endpoint_id)
    log_audit_event(
        db,
        request=request,
        tenant_id=workspace.id,
        action="webhook_endpoint.deleted",
        resource_type="webhook_endpoint",
        resource_id=endpoint_id,
    )


# ── POST /webhooks/{endpoint_id}/test ────────────────────────────────────────

@router.post("/{endpoint_id}/test", response_model=WebhookDeliveryOut)
async def test_webhook_endpoint(
    endpoint_id: uuid.UUID,
    workspace: Workspace = Depends(get_workspace),
    svc=Depends(_webhook_service_write),
) -> WebhookDeliveryOut:
    """Send a test ping to the endpoint and return the delivery result."""
    delivery = await svc.send_test_ping(workspace.id, endpoint_id)
    return WebhookDeliveryOut.model_validate(delivery)


# ── GET /webhooks/{endpoint_id}/deliveries ────────────────────────────────────

@router.get("/{endpoint_id}/deliveries", response_model=PaginatedWebhookDeliveries)
def list_webhook_deliveries(
    endpoint_id: uuid.UUID,
    page: int = 1,
    page_size: int = 20,
    workspace: Workspace = Depends(get_workspace),
    svc=Depends(_webhook_service),
) -> PaginatedWebhookDeliveries:
    """Return paginated delivery log for an endpoint."""
    return svc.list_deliveries(workspace.id, endpoint_id, page=page, page_size=page_size)
