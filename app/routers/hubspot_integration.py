"""
HubSpot CRM OAuth integration endpoints.

GET    /connect        — redirect to HubSpot's OAuth consent page (tenant-authenticated)
GET    /callback       — public; HubSpot redirects the browser here with no auth headers,
                          so the connecting tenant is recovered from the signed `state` param
DELETE ""              — revoke at HubSpot and delete the local connection
GET    ""              — connection status, settings toggles, and field mappings
GET    /contact        — CRM Search API lookup by phone, used by the dashboard and tests
PUT    /field-mapping  — save HubSpot field -> agent prompt-variable mappings
                          (validated against the live HubSpot contact schema)
POST   /field-mapping  — deprecated alias for PUT, kept for backward compatibility
PUT    /settings       — toggle contact-lookup / write-back for this workspace
GET    /sync-status    — last lookup/write-back times, status, rolling 24h error count
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_tenant, require_admin
from app.core.config import settings
from app.core.logger import logger
from app.schemas.base import SuccessResponse
from app.schemas.hubspot_integration import (
    HubSpotContactOut,
    HubSpotDisconnectResponse,
    HubSpotFieldMappingRequest,
    HubSpotFieldMappingResponse,
    HubSpotIntegrationStatusOut,
    HubSpotSettingsUpdateRequest,
    HubSpotSyncStatusOut,
)
from app.services import hubspot_service
from app.utils.response import create_success_response

router = APIRouter()


def _tenant_id(principal) -> uuid.UUID:
    return principal.current_tenant_id


@router.get("/connect",include_in_schema=False)
async def hubspot_connect(
    principal=Depends(require_admin),
):
    """Redirect to HubSpot's OAuth consent page. Scopes: contacts read/write."""
    tenant_id = _tenant_id(principal)
    state = hubspot_service.build_oauth_state(tenant_id)
    auth_url = hubspot_service.build_authorization_url(state)
    return RedirectResponse(url=auth_url, status_code=status.HTTP_302_FOUND)


@router.get("/callback",include_in_schema=False)
async def hubspot_callback(
    code: str = Query(...),
    state: str = Query(...),
    db: Session = Depends(get_db),
):
    """
    HubSpot OAuth callback. No auth dependency — this is a top-level browser
    redirect from HubSpot, which cannot carry our JWT/API-key headers. The
    connecting tenant is recovered from the signed `state` param instead.
    """
    try:
        tenant_id = hubspot_service.verify_oauth_state(state)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    try:
        token_response = await hubspot_service.exchange_code_for_tokens(code)
    except Exception as exc:
        logger.warning("HubSpot OAuth code exchange failed for tenant=%s: %s", tenant_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to exchange authorization code with HubSpot",
        )

    hubspot_service.upsert_tokens(db, tenant_id, token_response)

    base = (settings.FRONTEND_URL or "").rstrip("/")
    redirect_to = f"{base}/settings/integrations?hubspot=connected" if base else "/settings/integrations"
    return RedirectResponse(url=redirect_to, status_code=status.HTTP_302_FOUND)


@router.delete("", response_model=SuccessResponse[HubSpotDisconnectResponse])
async def hubspot_disconnect(
    principal=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Revoke the token at HubSpot and delete the workspaceintegration row."""
    tenant_id = _tenant_id(principal)
    disconnected = await hubspot_service.disconnect(db, tenant_id)
    if not disconnected:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="HubSpot is not connected for this workspace",
        )
    return create_success_response(
        HubSpotDisconnectResponse(disconnected=True),
        "HubSpot disconnected successfully",
    )


@router.get("", response_model=SuccessResponse[HubSpotIntegrationStatusOut])
async def hubspot_get_integration_status(
    principal=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Connection status, contact-lookup/write-back toggles, and field mappings for this workspace."""
    tenant_id = _tenant_id(principal)
    integration_settings = hubspot_service.get_integration_settings(db, tenant_id)
    return create_success_response(HubSpotIntegrationStatusOut(**integration_settings))


async def _validate_and_save_field_mapping(
    payload: HubSpotFieldMappingRequest, principal, db: Session
):
    """Shared handler for PUT and (deprecated) POST /field-mapping."""
    tenant_id = _tenant_id(principal)
    if not hubspot_service.tenant_has_hubspot_connected(db, tenant_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="HubSpot is not connected for this workspace",
        )

    try:
        valid_properties = set(
            await hubspot_service.get_hubspot_contact_properties(db, tenant_id)
        )
    except Exception as exc:
        logger.warning(
            "HubSpot contact properties fetch failed for tenant=%s: %s", tenant_id, exc
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to fetch HubSpot contact schema for field-mapping validation",
        )

    invalid_fields = sorted(
        {m.hubspot_field for m in payload.mappings if m.hubspot_field not in valid_properties}
    )
    if invalid_fields:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": f"Invalid HubSpot field names: {invalid_fields}",
                "invalid_fields": invalid_fields,
            },
        )

    mappings = [m.model_dump() for m in payload.mappings]
    hubspot_service.save_field_mappings(db, tenant_id, mappings)
    return create_success_response(
        HubSpotFieldMappingResponse(field_mappings=payload.mappings),
        "Field mappings saved successfully",
    )


@router.put("/field-mapping", response_model=SuccessResponse[HubSpotFieldMappingResponse])
async def hubspot_save_field_mapping(
    payload: HubSpotFieldMappingRequest,
    principal=Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Save the HubSpot field -> agent prompt-variable mappings for this workspace.

    Every hubspot_field must exist in the tenant's live HubSpot contact schema;
    unknown fields return 422 with the list of invalid field names.
    """
    return await _validate_and_save_field_mapping(payload, principal, db)


@router.post(
    "/field-mapping",
    response_model=SuccessResponse[HubSpotFieldMappingResponse],
    include_in_schema=False,
)
async def hubspot_save_field_mapping_legacy(
    payload: HubSpotFieldMappingRequest,
    principal=Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Deprecated alias for PUT /field-mapping, kept for backward compatibility."""
    return await _validate_and_save_field_mapping(payload, principal, db)


@router.put("/settings", response_model=SuccessResponse[HubSpotIntegrationStatusOut])
async def hubspot_update_settings(
    payload: HubSpotSettingsUpdateRequest,
    principal=Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Toggle contact-lookup and post-call write-back for this workspace."""
    tenant_id = _tenant_id(principal)
    if not hubspot_service.tenant_has_hubspot_connected(db, tenant_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="HubSpot is not connected for this workspace",
        )

    hubspot_service.update_integration_settings(
        db,
        tenant_id,
        contact_lookup_enabled=payload.contact_lookup_enabled,
        write_back_enabled=payload.write_back_enabled,
    )
    integration_settings = hubspot_service.get_integration_settings(db, tenant_id)
    return create_success_response(
        HubSpotIntegrationStatusOut(**integration_settings),
        "Settings updated successfully",
    )


@router.get("/sync-status", response_model=SuccessResponse[HubSpotSyncStatusOut])
async def hubspot_sync_status(
    principal=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Last contact-lookup/write-back times, write-back status, and rolling 24h error count."""
    tenant_id = _tenant_id(principal)
    sync_status = hubspot_service.get_sync_status(db, tenant_id)
    return create_success_response(HubSpotSyncStatusOut(**sync_status))


@router.get("/contact", response_model=SuccessResponse[HubSpotContactOut])
async def hubspot_get_contact(
    phone: str = Query(..., description="Phone number to search for (E.164 or local format)"),
    principal=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """CRM Search API lookup by phone. Redis-cached for 5 minutes per phone number."""
    tenant_id = _tenant_id(principal)

    if not hubspot_service.tenant_has_hubspot_connected(db, tenant_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="HubSpot is not connected for this workspace",
        )

    contact = await hubspot_service.get_contact_for_phone(db, tenant_id, phone)
    if not contact:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No HubSpot contact found for this phone number",
        )

    return create_success_response(HubSpotContactOut(**contact))
