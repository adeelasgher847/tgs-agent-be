"""
Calendly calendar integration — OAuth endpoints + availability/booking proxy.

GET    /integrations/calendly/connect   — redirect to Calendly's OAuth consent page (tenant-authenticated)
GET    /integrations/calendly/callback  — public; Calendly redirects the browser here with no auth headers,
                                           so the connecting workspace is recovered from the signed `state` param
GET    /integrations/calendly           — connection status: {connected, user_uri, event_type_uri}
DELETE /integrations/calendly           — revoke access token at Calendly and delete the local row
GET    /calendar/availability           — bookable slots for the connected event type
POST   /calendar/events                 — schedule an appointment on Calendly
"""
from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin_or_api_key, require_tenant
from app.core.config import settings
from app.core.logger import logger
from app.schemas.base import SuccessResponse
from app.schemas.calendly_integration import CalendlyDisconnectResponse, CalendlyIntegrationStatusOut
from app.services import calendly_service
from app.utils.response import create_success_response

router = APIRouter(prefix="/integrations/calendly", tags=["Calendly Integration"])
calendar_router = APIRouter(prefix="/calendar", tags=["Calendly Calendar"])


def _workspace_id(principal) -> uuid.UUID:
    return principal.current_tenant_id


@router.get("/connect", include_in_schema=False)
async def calendly_connect(
    principal=Depends(require_admin_or_api_key),
):
    """Redirect to Calendly's OAuth consent page."""
    workspace_id = _workspace_id(principal)
    state = calendly_service.build_oauth_state(workspace_id)
    auth_url = calendly_service.build_authorization_url(state)
    return RedirectResponse(url=auth_url, status_code=status.HTTP_302_FOUND)


@router.get("/callback", include_in_schema=False)
async def calendly_callback(
    code: str = Query(...),
    state: str = Query(...),
    db: Session = Depends(get_db),
):
    """
    Calendly OAuth callback. No auth dependency — this is a top-level browser
    redirect from Calendly, which cannot carry our JWT/API-key headers. The
    connecting workspace is recovered from the signed `state` param instead.
    """
    try:
        workspace_id = calendly_service.verify_oauth_state(state)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    try:
        token_response = await calendly_service.exchange_code_for_tokens(code)
    except Exception as exc:
        logger.warning("Calendly OAuth code exchange failed for workspace=%s: %s", workspace_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to exchange authorization code with Calendly",
        )

    calendly_user_uri = None
    try:
        current_user = await calendly_service.get_current_user(token_response["access_token"])
        calendly_user_uri = (current_user.get("resource") or {}).get("uri")
    except Exception:
        logger.warning(
            "Calendly get_current_user failed for workspace=%s (continuing without user_uri)",
            workspace_id,
            exc_info=True,
        )

    calendly_service.upsert_tokens(
        db,
        workspace_id,
        token_response,
        calendly_user_uri=calendly_user_uri,
    )

    base = (settings.FRONTEND_URL or "").rstrip("/")
    redirect_to = f"{base}/settings/integrations?calendly=connected" if base else "/settings/integrations"
    return RedirectResponse(url=redirect_to, status_code=status.HTTP_302_FOUND)


@router.get("", response_model=SuccessResponse[CalendlyIntegrationStatusOut])
async def calendly_get_integration_status(
    principal=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Connection status for this workspace."""
    workspace_id = _workspace_id(principal)
    row = calendly_service.get_integration(db, workspace_id)
    if row is None:
        return create_success_response(CalendlyIntegrationStatusOut(connected=False))
    return create_success_response(
        CalendlyIntegrationStatusOut(
            connected=True,
            user_uri=row.calendly_user_uri,
            event_type_uri=row.calendly_event_type_uri,
        )
    )


class CalendlyEventTypeUpdateRequest(BaseModel):
    event_type_uri: str


@router.put("/event-type", response_model=SuccessResponse[CalendlyIntegrationStatusOut])
async def calendly_set_event_type(
    payload: CalendlyEventTypeUpdateRequest,
    principal=Depends(require_admin_or_api_key),
    db: Session = Depends(get_db),
):
    """Select which Calendly event type to check availability for / book against."""
    workspace_id = _workspace_id(principal)
    row = calendly_service.get_integration(db, workspace_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Calendly is not connected for this workspace",
        )
    row.calendly_event_type_uri = payload.event_type_uri
    db.add(row)
    db.commit()
    db.refresh(row)
    return create_success_response(
        CalendlyIntegrationStatusOut(
            connected=True,
            user_uri=row.calendly_user_uri,
            event_type_uri=row.calendly_event_type_uri,
        )
    )


@router.delete("", response_model=SuccessResponse[CalendlyDisconnectResponse])
async def calendly_disconnect(
    principal=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Revoke the token at Calendly and delete the calendlyintegration row."""
    workspace_id = _workspace_id(principal)
    disconnected = await calendly_service.disconnect(db, workspace_id)
    if not disconnected:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Calendly is not connected for this workspace",
        )
    return create_success_response(
        CalendlyDisconnectResponse(disconnected=True),
        "Calendly disconnected successfully",
    )


# ─── Availability / booking proxy ──────────────────────────────────────────────


class CalendlySlotOut(BaseModel):
    slot_start: datetime
    slot_end: str | None = None
    available: bool


class CalendlyAvailabilityResponse(BaseModel):
    slots: list[CalendlySlotOut]


class CalendlyBookEventRequest(BaseModel):
    start_time: datetime
    attendee_email: str
    attendee_name: str
    description: str | None = None


@calendar_router.get("/availability", response_model=SuccessResponse[CalendlyAvailabilityResponse])
async def calendly_get_availability(
    date_from: datetime = Query(...),
    date_to: datetime = Query(...),
    principal=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    workspace_id = _workspace_id(principal)
    try:
        slots = await calendly_service.get_available_slots(db, workspace_id, date_from, date_to)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.warning("Calendly availability lookup failed for workspace=%s: %s", workspace_id, exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Calendly availability lookup failed")
    return create_success_response(
        CalendlyAvailabilityResponse(slots=[CalendlySlotOut(**s) for s in slots])
    )


@calendar_router.post("/events", status_code=status.HTTP_201_CREATED)
async def calendly_create_event(
    payload: CalendlyBookEventRequest,
    principal=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    workspace_id = _workspace_id(principal)
    try:
        result = await calendly_service.book_appointment(
            db,
            workspace_id,
            start_time=payload.start_time,
            attendee_email=payload.attendee_email,
            attendee_name=payload.attendee_name,
            description=payload.description,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.warning("Calendly booking failed for workspace=%s: %s", workspace_id, exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Calendly booking failed")
    return create_success_response(data=result, status_code=status.HTTP_201_CREATED)
