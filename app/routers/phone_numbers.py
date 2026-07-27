"""
Phone Numbers router — /api/v1/phone-numbers

Sprint 2 additions:
  GET  /search    — search available Twilio numbers
  POST /purchase  — purchase + persist atomically

Legacy routes kept:
  GET  /           list (upgraded to binding-aware response)
  POST /           create (env-creds based; kept for backward compat)
  POST /import     BYO import with per-number creds
  GET, PUT, DELETE /{id}
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin, require_config, require_readonly
from app.models.user import User
from app.schemas.base import SuccessResponse
from app.schemas.phone_number import (
    CreatePhoneNumberRequest,
    CreatePhoneNumberResponse,
    ImportTwilioPhoneNumberRequest,
    ImportTwilioPhoneNumberResponse,
    PhoneNumberResponse,
    PhoneNumberUpdate,
    PhoneNumberWithBinding,
    PhoneNumberWithBindingList,
    PurchasePhoneNumberRequest,
    PurchasePhoneNumberResponse,
    PhoneNumberSearchResponse,
    NumberConfigurationRequest,
    NumberConfigurationResponse,
)
from app.services.phone_number_service import phone_number_service
from app.services.twilio_service import twilio_service
from app.utils.response import create_success_response
from app.core.logger import logger

router = APIRouter()


# ---------------------------------------------------------------------------
# Search available numbers — GET /search
# Ticket: GET /api/v1/phone-numbers/search?country=AU&type=local&areaCode=02
# ---------------------------------------------------------------------------


@router.get("/search", response_model=SuccessResponse[PhoneNumberSearchResponse])
async def search_phone_numbers(
    country: str = Query(default="AU", description="ISO country code e.g. AU, US, GB"),
    type: str = Query(default="local", description="Number type: local | toll_free | mobile"),
    areaCode: Optional[str] = Query(default=None, description="Area code to filter by e.g. 02"),
    limit: int = Query(default=20, ge=1, le=100),
    user: User = Depends(require_readonly),
) -> SuccessResponse[PhoneNumberSearchResponse]:
    """Search available Twilio phone numbers by country, type and area code."""
    try:
        results = twilio_service.search_available_numbers(
            country_code=country,
            number_type=type,
            area_code=areaCode,
            limit=limit,
        )
    except Exception as exc:
        logger.error("Twilio number search failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=502,
            detail="Phone number search temporarily unavailable. Please try again.",
        )

    return create_success_response(
        PhoneNumberSearchResponse(available_numbers=results, total=len(results)),
        f"Found {len(results)} available numbers",
    )


# ---------------------------------------------------------------------------
# Purchase — POST /purchase
# Ticket: POST /api/v1/phone-numbers/purchase
# ---------------------------------------------------------------------------


@router.post("/purchase", response_model=SuccessResponse[PurchasePhoneNumberResponse])
async def purchase_phone_number(
    request: PurchasePhoneNumberRequest,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> SuccessResponse[PurchasePhoneNumberResponse]:
    """
    Purchase a phone number from Twilio and persist a DB row atomically.

    Staging (ENVIRONMENT=staging): uses Twilio test credentials — no real purchase.
    """
    pn = phone_number_service.purchase_phone_number(
        db=db,
        phone_number=request.phone_number,
        tenant_id=user.current_tenant_id,
        label=request.label,
    )
    return create_success_response(
        PurchasePhoneNumberResponse(
            id=pn.id,
            phone_number=pn.phone_number,
            provider=pn.provider,
            twilio_sid=pn.twilio_phone_number_sid,
            status=pn.status,
            workspace_id=pn.tenant_id,
            created_at=pn.created_at,
            message="Phone number purchased and registered successfully",
        ),
        "Phone number purchased",
    )


# ---------------------------------------------------------------------------
# List — GET / (upgraded: includes binding status + agent name)
# ---------------------------------------------------------------------------


@router.get("/", response_model=SuccessResponse[PhoneNumberWithBindingList])
async def get_phone_numbers(
    user: User = Depends(require_readonly),
    db: Session = Depends(get_db),
) -> SuccessResponse[PhoneNumberWithBindingList]:
    """List all phone numbers for the workspace with binding status and agent name."""
    numbers = phone_number_service.list_numbers_with_binding(db, user.current_tenant_id)
    items = [PhoneNumberWithBinding(**n) for n in numbers]
    return create_success_response(
        PhoneNumberWithBindingList(phone_numbers=items, total=len(items)),
        f"Retrieved {len(items)} phone numbers",
    )


# ---------------------------------------------------------------------------
# Create (legacy env-creds path) — POST /
# ---------------------------------------------------------------------------


@router.post("/", response_model=SuccessResponse[CreatePhoneNumberResponse])
async def create_phone_number(
    request: CreatePhoneNumberRequest,
    user: User = Depends(require_config),
    db: Session = Depends(get_db),
) -> SuccessResponse[CreatePhoneNumberResponse]:
    """Register a number already in the platform's Twilio account."""
    from app.models.agent import Agent
    from app.schemas.phone_number import PhoneNumberCreate

    if request.agent_id:
        from sqlalchemy import select as sa_select

        agent = db.execute(
            sa_select(Agent).where(
                Agent.id == request.agent_id,
                Agent.tenant_id == user.current_tenant_id,
            )
        ).scalar_one_or_none()
        if not agent:
            raise HTTPException(
                status_code=400,
                detail=f"Agent {request.agent_id} not found",
            )

    try:
        pn = phone_number_service.create_phone_number(
            db,
            PhoneNumberCreate(
                phone_number=request.phone_number,
                label=request.label,
                assistant_id=request.agent_id,
                tenant_id=user.current_tenant_id,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return create_success_response(
        CreatePhoneNumberResponse(
            id=pn.id,
            phone_number=pn.phone_number,
            label=pn.label,
            status=pn.status,
            created_at=pn.created_at,
            message="Phone number created successfully",
        ),
        "Phone number created",
    )


# ---------------------------------------------------------------------------
# Import (legacy BYO with per-number creds) — POST /import
# ---------------------------------------------------------------------------


@router.post("/import", response_model=SuccessResponse[ImportTwilioPhoneNumberResponse])
async def import_twilio_phone_number(
    request: ImportTwilioPhoneNumberRequest,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> SuccessResponse[ImportTwilioPhoneNumberResponse]:
    """Import a Twilio number with custom Account SID / Auth Token (BYO Twilio account)."""
    try:
        pn = phone_number_service.import_twilio_phone_number(
            db=db,
            phone_number=request.phone_number,
            label=request.label,
            tenant_id=user.current_tenant_id,
            twilio_account_sid=request.twilio_account_sid,
            twilio_auth_token=request.twilio_auth_token,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return create_success_response(
        ImportTwilioPhoneNumberResponse(
            id=pn.id,
            phone_number=pn.phone_number,
            label=pn.label,
            status=pn.status,
            twilio_account_sid="***encrypted***",
            created_at=pn.created_at,
            message="Twilio phone number imported successfully",
        ),
        "Phone number imported",
    )


# ---------------------------------------------------------------------------
# Legacy hidden routes (kept for backward compat; not in OpenAPI docs)
# ---------------------------------------------------------------------------


@router.get("/available-numbers", include_in_schema=False)
async def get_available_phone_numbers_legacy(
    country_code: str = Query(default="US"),
    area_code: Optional[str] = Query(default=None),
    contains: Optional[str] = Query(default=None),
    voice_enabled: bool = Query(default=True),
    sms_enabled: bool = Query(default=True),
    limit: int = Query(default=20, ge=1, le=100),
    user: User = Depends(require_readonly),
):
    """Legacy search endpoint — use GET /search instead."""
    try:
        results = twilio_service.search_available_numbers(
            country_code=country_code,
            area_code=area_code,
            contains=contains,
            voice_enabled=voice_enabled,
            sms_enabled=sms_enabled,
            limit=limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return create_success_response(
        {"available_numbers": results, "total": len(results)},
        f"Found {len(results)} available numbers",
    )


@router.post("/twilio/purchase", include_in_schema=False)
async def purchase_phone_number_legacy(
    phone_number: str = Query(...),
    webhook_url: Optional[str] = Query(default=None),
    status_callback_url: Optional[str] = Query(default=None),
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Legacy purchase endpoint — use POST /purchase instead (this route now saves to DB)."""
    from app.core.config import settings

    if not webhook_url:
        webhook_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/incoming"
    if not status_callback_url:
        status_callback_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/call-events"

    pn = phone_number_service.purchase_phone_number(
        db=db,
        phone_number=phone_number,
        tenant_id=user.current_tenant_id,
    )
    return create_success_response(
        {"id": str(pn.id), "phone_number": pn.phone_number, "twilio_sid": pn.twilio_phone_number_sid},
        f"Phone number {phone_number} purchased and saved",
    )


@router.get("/twilio/account-info", include_in_schema=False)
async def get_twilio_account_info(user: User = Depends(require_readonly)):
    try:
        return create_success_response(
            {"account_info": twilio_service.get_account_info()},
            "Account info retrieved",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/available-number", include_in_schema=False)
async def get_owned_phone_numbers(
    limit: int = Query(default=50, ge=1, le=100),
    user: User = Depends(require_readonly),
):
    try:
        owned = twilio_service.list_owned_numbers(limit=limit)
        return create_success_response(
            {"owned_numbers": owned, "total": len(owned)}, f"Found {len(owned)} owned numbers"
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Detail / update / delete — /{phone_number_id}
# ---------------------------------------------------------------------------


@router.get("/{phone_number_id}", response_model=SuccessResponse[PhoneNumberResponse])
async def get_phone_number(
    phone_number_id: uuid.UUID,
    user: User = Depends(require_readonly),
    db: Session = Depends(get_db),
) -> SuccessResponse[PhoneNumberResponse]:
    pn = phone_number_service.get_phone_number_by_id(db, phone_number_id, user.current_tenant_id)
    if not pn:
        raise HTTPException(status_code=404, detail="Phone number not found")
    return create_success_response(
        PhoneNumberResponse.model_validate(pn), "Phone number retrieved"
    )


@router.put("/{phone_number_id}", response_model=SuccessResponse[PhoneNumberResponse])
async def update_phone_number(
    phone_number_id: uuid.UUID,
    request: PhoneNumberUpdate,
    user: User = Depends(require_config),
    db: Session = Depends(get_db),
) -> SuccessResponse[PhoneNumberResponse]:
    pn = phone_number_service.update_phone_number(
        db, phone_number_id, user.current_tenant_id, request
    )
    if not pn:
        raise HTTPException(status_code=404, detail="Phone number not found")
    return create_success_response(PhoneNumberResponse.model_validate(pn), "Phone number updated")


@router.delete("/{phone_number_id}", response_model=SuccessResponse[dict])
async def delete_phone_number(
    phone_number_id: uuid.UUID,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> SuccessResponse[dict]:
    ok = phone_number_service.delete_phone_number(db, phone_number_id, user.current_tenant_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Phone number not found")
    return create_success_response({"deleted": True}, "Phone number deleted")


# ---------------------------------------------------------------------------
# Number Configuration — /{phone_number_id}/configuration
# PUT  upsert recording_enabled, max_duration_seconds, business_hours
# GET  read current configuration
# ---------------------------------------------------------------------------


@router.put(
    "/{phone_number_id}/configuration",
    response_model=SuccessResponse[NumberConfigurationResponse],
)
async def upsert_number_configuration(
    phone_number_id: uuid.UUID,
    request: NumberConfigurationRequest,
    user: User = Depends(require_config),
    db: Session = Depends(get_db),
) -> SuccessResponse[NumberConfigurationResponse]:
    """Update or create the per-number configuration (recording, duration, hours)."""
    config = phone_number_service.upsert_number_configuration(
        db=db,
        phone_number_id=phone_number_id,
        tenant_id=user.current_tenant_id,
        recording_enabled=request.recording_enabled,
        max_duration_seconds=request.max_duration_seconds,
        business_hours=request.business_hours.model_dump() if request.business_hours else None,
    )
    return create_success_response(
        NumberConfigurationResponse.model_validate(config),
        "Number configuration updated",
    )


@router.get(
    "/{phone_number_id}/configuration",
    response_model=SuccessResponse[NumberConfigurationResponse],
)
async def get_number_configuration(
    phone_number_id: uuid.UUID,
    user: User = Depends(require_readonly),
    db: Session = Depends(get_db),
) -> SuccessResponse[NumberConfigurationResponse]:
    """Retrieve the per-number configuration."""
    pn = phone_number_service._require_number(db, phone_number_id, user.current_tenant_id)
    config = pn.configuration
    if config is None:
        raise HTTPException(status_code=404, detail="No configuration found for this number")
    return create_success_response(
        NumberConfigurationResponse.model_validate(config),
        "Number configuration retrieved",
    )
