"""
GoHighLevel (GHL) CRM OAuth integration endpoints.

GET    /connect        — redirect to GHL's OAuth consent page (tenant-authenticated)
GET    /callback       — public; GHL redirects the browser here with no auth headers,
                          so the connecting tenant is recovered from the signed `state` param
DELETE ""              — revoke local credentials and delete the connection
GET    ""              — connection status and write-back toggle
GET    /contact        — Contacts API lookup by phone, used by the dashboard and tests
POST   /note           — create a note directly against a GHL contact
PUT    /settings        — toggle post-call write-back for this workspace
GET    /sync-status    — last lookup/write-back times, status, rolling 24h error count
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_tenant, require_admin, require_manager
from app.core.config import settings
from app.core.logger import logger
from app.schemas.base import SuccessResponse
from app.schemas.ghl_integration import (
    GhlContactOut,
    GhlDisconnectResponse,
    GhlIntegrationStatusOut,
    GhlNoteCreateRequest,
    GhlNoteCreateResponse,
    GhlSettingsUpdateRequest,
    GhlSyncStatusOut,
)
from app.services import ghl_service
from app.utils.response import create_success_response

router = APIRouter()


def _tenant_id(principal) -> uuid.UUID:
    return principal.current_tenant_id


@router.get("/connect", include_in_schema=False)
async def ghl_connect(
    principal=Depends(require_admin),
):
    """Redirect to GHL's OAuth consent page. Scopes: contacts.readonly, contacts.write."""
    tenant_id = _tenant_id(principal)
    state = ghl_service.build_oauth_state(tenant_id)
    auth_url = ghl_service.build_authorization_url(state)
    return RedirectResponse(url=auth_url, status_code=status.HTTP_302_FOUND)


@router.get("/callback", include_in_schema=False)
async def ghl_callback(
    code: str = Query(...),
    state: str = Query(...),
    db: Session = Depends(get_db),
):
    """
    GHL OAuth callback. No auth dependency — this is a top-level browser
    redirect from GHL, which cannot carry our JWT/API-key headers. The
    connecting tenant is recovered from the signed `state` param instead.
    """
    try:
        tenant_id = ghl_service.verify_oauth_state(state)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    try:
        token_response = await ghl_service.exchange_code_for_tokens(code)
    except Exception as exc:
        logger.warning("GHL OAuth code exchange failed for tenant=%s: %s", tenant_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to exchange authorization code with GoHighLevel",
        )

    ghl_service.upsert_tokens(db, tenant_id, token_response)

    base = (settings.FRONTEND_URL or "").rstrip("/")
    redirect_to = f"{base}/settings/integrations?ghl=connected" if base else "/settings/integrations"
    return RedirectResponse(url=redirect_to, status_code=status.HTTP_302_FOUND)


@router.delete("", response_model=SuccessResponse[GhlDisconnectResponse])
async def ghl_disconnect(
    principal=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Revoke local GHL OAuth credentials and delete the workspaceintegration row."""
    tenant_id = _tenant_id(principal)
    disconnected = await ghl_service.disconnect(db, tenant_id)
    if not disconnected:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="GoHighLevel is not connected for this workspace",
        )
    return create_success_response(
        GhlDisconnectResponse(disconnected=True),
        "GoHighLevel disconnected successfully",
    )


@router.get("", response_model=SuccessResponse[GhlIntegrationStatusOut])
async def ghl_get_integration_status(
    principal=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Connection status and write-back toggle for this workspace."""
    tenant_id = _tenant_id(principal)
    integration_settings = ghl_service.get_integration_settings(db, tenant_id)
    return create_success_response(GhlIntegrationStatusOut(**integration_settings))


@router.put("/settings", response_model=SuccessResponse[GhlIntegrationStatusOut])
async def ghl_update_settings(
    payload: GhlSettingsUpdateRequest,
    principal=Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Toggle post-call write-back for this workspace."""
    tenant_id = _tenant_id(principal)
    if not ghl_service.tenant_has_ghl_connected(db, tenant_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GoHighLevel is not connected for this workspace",
        )

    ghl_service.update_integration_settings(
        db, tenant_id, write_back_enabled=payload.write_back_enabled
    )
    integration_settings = ghl_service.get_integration_settings(db, tenant_id)
    return create_success_response(
        GhlIntegrationStatusOut(**integration_settings),
        "Settings updated successfully",
    )


@router.get("/sync-status", response_model=SuccessResponse[GhlSyncStatusOut])
async def ghl_sync_status(
    principal=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Last contact-lookup/write-back times, write-back status, and rolling 24h error count."""
    tenant_id = _tenant_id(principal)
    sync_status = ghl_service.get_sync_status(db, tenant_id)
    return create_success_response(GhlSyncStatusOut(**sync_status))


@router.get("/contact", response_model=SuccessResponse[GhlContactOut])
async def ghl_get_contact(
    phone: str = Query(..., description="Phone number to search for (E.164 or local format)"),
    principal=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Contacts API lookup by phone. Redis-cached for 5 minutes per phone number."""
    tenant_id = _tenant_id(principal)

    if not ghl_service.tenant_has_ghl_connected(db, tenant_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GoHighLevel is not connected for this workspace",
        )

    contact = await ghl_service.get_contact_for_phone(db, tenant_id, phone)
    if not contact:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No GoHighLevel contact found for this phone number",
        )

    return create_success_response(GhlContactOut(**contact))


@router.post("/note", response_model=SuccessResponse[GhlNoteCreateResponse])
async def ghl_create_note(
    payload: GhlNoteCreateRequest,
    principal=Depends(require_manager),
    db: Session = Depends(get_db),
):
    """Create a note directly against a GHL contact. Writes to the tenant's live
    CRM, so (like /settings) this requires manager+ rather than any authenticated
    member — a read_only member must not be able to write into GoHighLevel."""
    tenant_id = _tenant_id(principal)

    if not ghl_service.tenant_has_ghl_connected(db, tenant_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GoHighLevel is not connected for this workspace",
        )

    token_info = await ghl_service.get_valid_access_token(db, tenant_id)
    if not token_info:
        # Connected, but the stored token is unusable (refresh failed/expired
        # refresh token) — distinct from "never connected" so the client
        # doesn't wrongly prompt the user to redo the OAuth flow for what may
        # be a transient GHL outage.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to obtain a valid GoHighLevel access token",
        )
    access_token, _location_id = token_info

    try:
        note_response = await ghl_service.create_note(
            access_token, payload.contact_id, payload.content, tenant_id
        )
    except Exception as exc:
        logger.warning(
            "GHL note creation failed for tenant=%s contact=%s: %s",
            tenant_id,
            payload.contact_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to create note in GoHighLevel",
        )

    note_id = note_response.get("id") or (note_response.get("note") or {}).get("id")
    return create_success_response(
        GhlNoteCreateResponse(id=note_id, contact_id=payload.contact_id),
        "Note created successfully",
    )
