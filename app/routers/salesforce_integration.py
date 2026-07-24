"""
Salesforce CRM OAuth integration endpoints.

GET    /connect        — redirect to Salesforce's OAuth consent page (tenant-authenticated)
GET    /callback       — public; Salesforce redirects the browser here with no auth headers,
                          so the connecting tenant is recovered from the signed `state` param
DELETE ""              — revoke at Salesforce and delete the local connection
GET    ""              — connection status and write-back toggle
GET    /contact        — SOQL contact lookup by phone, used by the dashboard and tests
PUT    /settings        — toggle post-call write-back for this workspace
GET    /sync-status    — last lookup/write-back times, status, rolling 24h error count
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_tenant, require_admin
from app.core.config import settings
from app.core.logger import logger
from app.schemas.base import SuccessResponse
from app.schemas.salesforce_integration import (
    SalesforceContactOut,
    SalesforceDisconnectResponse,
    SalesforceIntegrationStatusOut,
    SalesforceSettingsUpdateRequest,
    SalesforceSyncStatusOut,
)
from app.services import salesforce_service
from app.utils.response import create_success_response

router = APIRouter()


def _tenant_id(principal) -> uuid.UUID:
    return principal.current_tenant_id


@router.get("/connect", include_in_schema=False)
async def salesforce_connect(
    principal=Depends(require_admin),
):
    """Redirect to Salesforce's OAuth consent page. Scopes: api, refresh_token."""
    tenant_id = _tenant_id(principal)
    state = salesforce_service.build_oauth_state(tenant_id)
    auth_url = salesforce_service.build_authorization_url(state)
    return RedirectResponse(url=auth_url, status_code=status.HTTP_302_FOUND)


@router.get("/callback", include_in_schema=False)
async def salesforce_callback(
    code: str = Query(...),
    state: str = Query(...),
    db: Session = Depends(get_db),
):
    """
    Salesforce OAuth callback. No auth dependency — this is a top-level browser
    redirect from Salesforce, which cannot carry our JWT/API-key headers. The
    connecting tenant is recovered from the signed `state` param instead.
    """
    try:
        tenant_id = salesforce_service.verify_oauth_state(state)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    try:
        token_response = await salesforce_service.exchange_code_for_tokens(code)
    except Exception as exc:
        logger.warning("Salesforce OAuth code exchange failed for tenant=%s: %s", tenant_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to exchange authorization code with Salesforce",
        )

    salesforce_service.upsert_tokens(db, tenant_id, token_response)

    base = (settings.FRONTEND_URL or "").rstrip("/")
    redirect_to = f"{base}/settings/integrations?salesforce=connected" if base else "/settings/integrations"
    return RedirectResponse(url=redirect_to, status_code=status.HTTP_302_FOUND)


@router.delete("", response_model=SuccessResponse[SalesforceDisconnectResponse])
async def salesforce_disconnect(
    principal=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Revoke the token at Salesforce and delete the workspaceintegration row."""
    tenant_id = _tenant_id(principal)
    disconnected = await salesforce_service.disconnect(db, tenant_id)
    if not disconnected:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Salesforce is not connected for this workspace",
        )
    return create_success_response(
        SalesforceDisconnectResponse(disconnected=True),
        "Salesforce disconnected successfully",
    )


@router.get("", response_model=SuccessResponse[SalesforceIntegrationStatusOut])
async def salesforce_get_integration_status(
    principal=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Connection status and write-back toggle for this workspace."""
    tenant_id = _tenant_id(principal)
    integration_settings = salesforce_service.get_integration_settings(db, tenant_id)
    return create_success_response(SalesforceIntegrationStatusOut(**integration_settings))


@router.put("/settings", response_model=SuccessResponse[SalesforceIntegrationStatusOut])
async def salesforce_update_settings(
    payload: SalesforceSettingsUpdateRequest,
    principal=Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Toggle post-call write-back for this workspace."""
    tenant_id = _tenant_id(principal)
    if not salesforce_service.tenant_has_salesforce_connected(db, tenant_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Salesforce is not connected for this workspace",
        )

    salesforce_service.update_integration_settings(
        db, tenant_id, write_back_enabled=payload.write_back_enabled
    )
    integration_settings = salesforce_service.get_integration_settings(db, tenant_id)
    return create_success_response(
        SalesforceIntegrationStatusOut(**integration_settings),
        "Settings updated successfully",
    )


@router.get("/sync-status", response_model=SuccessResponse[SalesforceSyncStatusOut])
async def salesforce_sync_status(
    principal=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Last contact-lookup/write-back times, write-back status, and rolling 24h error count."""
    tenant_id = _tenant_id(principal)
    sync_status = salesforce_service.get_sync_status(db, tenant_id)
    return create_success_response(SalesforceSyncStatusOut(**sync_status))


@router.get("/contact", response_model=SuccessResponse[SalesforceContactOut])
async def salesforce_get_contact(
    phone: str = Query(..., description="Phone number to search for (E.164 or local format)"),
    principal=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """SOQL contact lookup by phone. Redis-cached for 5 minutes per phone number."""
    tenant_id = _tenant_id(principal)

    if not salesforce_service.tenant_has_salesforce_connected(db, tenant_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Salesforce is not connected for this workspace",
        )

    contact = await salesforce_service.get_contact_for_phone(db, tenant_id, phone)
    if not contact:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No Salesforce contact found for this phone number",
        )

    return create_success_response(SalesforceContactOut(**contact))
