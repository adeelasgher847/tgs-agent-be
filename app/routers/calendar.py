from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from typing import Optional
from datetime import date
import uuid

from app.api.deps import get_db, require_tenant
from app.core.config import settings
from app.models.user import User
from app.schemas.base import SuccessResponse
from typing import List
from app.schemas.calendar import (
    AppointmentCreate, AppointmentStatusUpdate, AppointmentReschedule, AppointmentOut,
    AppointmentListItemOut,
    AppointmentDetailOut,
    AppointmentListResponse,
    AppointmentIntakeSummaryResponse,
    BusinessHoursUpsert, BusinessHoursOut,
)
from app.services.calendar_service import calendar_service
from app.services.business_hours_service import BusinessHoursConflictError, business_hours_service
from app.services.appointment_intake_summary_service import appointment_intake_summary_service
from app.utils.response import create_success_response

router = APIRouter()

# External API mapping (stable public contract):
# 0=Sunday ... 6=Saturday
# Internal storage/service mapping remains Python weekday:
# 0=Monday ... 6=Sunday
def _api_day_to_internal(day: int) -> int:
    return (day + 6) % 7


def _internal_day_to_api(day: int) -> int:
    return (day + 1) % 7


def _map_hours_payload_to_internal(payload: List[BusinessHoursUpsert]) -> List[BusinessHoursUpsert]:
    return [
        item.model_copy(update={"day_of_week": _api_day_to_internal(item.day_of_week)})
        for item in payload
    ]


def _map_hours_out_to_api(rows: List[BusinessHoursOut]) -> List[BusinessHoursOut]:
    return [
        row.model_copy(update={"day_of_week": _internal_day_to_api(row.day_of_week)})
        for row in rows
    ]


# ─── Appointments (local read-log; Calendly owns availability/booking) ────────

@router.post("/appointments", response_model=SuccessResponse[AppointmentOut], status_code=status.HTTP_201_CREATED)
def create_appointment(
    payload: AppointmentCreate,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Log an appointment (already scheduled on Calendly, or via legacy web flow)."""
    try:
        appt = calendar_service.book_appointment(
            db=db,
            tenant_id=user.current_tenant_id,
            customer_name=payload.customer_name,
            customer_phone=payload.customer_phone,
            slot_start=payload.slot_start,
            agent_id=payload.agent_id,
            appointment_reason=payload.appointment_reason,
            customer_email=payload.customer_email,
            notes=payload.notes,
            created_via="web",
            notify_user_id=user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return create_success_response(
        data=calendar_service.to_appointment_out(db, user.current_tenant_id, appt),
        status_code=status.HTTP_201_CREATED,
    )


@router.get("/appointments", response_model=SuccessResponse[AppointmentListResponse],include_in_schema=False)
def list_appointments(
    date_from: Optional[date] = Query(None),
    date_to: Optional[date] = Query(None),
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """List appointments filtered only by date range."""
    items, total = calendar_service.get_appointments(
        db=db,
        tenant_id=user.current_tenant_id,
        date_from=date_from,
        date_to=date_to,
    )
    return create_success_response(
        data=AppointmentListResponse(
            appointments=[
                AppointmentListItemOut(
                    id=full.id,
                    appointment_reason=full.appointment_reason,
                    slot_start_local=full.slot_start_local,
                    slot_end_local=full.slot_end_local,
                )
                for full in [
                    calendar_service.to_appointment_out(db, user.current_tenant_id, a)
                    for a in items
                ]
            ],
            total=total,
        )
    )


@router.get("/appointments/acknowledge", include_in_schema=False)
def acknowledge_appointment_review(
    token: str = Query(..., description="Signed acknowledgement token from email."),
    db: Session = Depends(get_db),
):
    """
    Public review-link endpoint (token-authenticated).
    Marks an appointment as reviewed and redirects to frontend appointments page.
    """
    try:
        appt = calendar_service.acknowledge_appointment_from_token(db=db, token=token)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    base = (settings.FRONTEND_URL or "").rstrip("/")
    redirect_to = (
        f"{base}/appointments?appointmentId={appt.id}&reviewed=1"
        if base
        else "/appointments"
    )
    return RedirectResponse(url=redirect_to, status_code=status.HTTP_302_FOUND)


@router.get(
    "/appointments/{appointment_id}/intake-summary",
    response_model=SuccessResponse[AppointmentIntakeSummaryResponse],
)
def get_appointment_intake_summary(
    appointment_id: uuid.UUID,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """
    Generate a fresh intake briefing from the call transcript linked to this appointment.
    Not stored — suitable for demo; each request runs a new LLM extraction.

    Omits sentiment scores, satisfaction metrics, and emotional analytics by design.
    """
    appt = calendar_service.get_appointment_by_id(db, appointment_id, user.current_tenant_id)
    if not appt:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Appointment not found")
    payload = appointment_intake_summary_service.generate_intake_summary(
        db=db,
        tenant_id=user.current_tenant_id,
        appointment=appt,
    )
    return create_success_response(
        data=AppointmentIntakeSummaryResponse.model_validate(payload),
    )


@router.get("/appointments/{appointment_id}", response_model=SuccessResponse[AppointmentDetailOut])
def get_appointment(
    appointment_id: uuid.UUID,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    appt = calendar_service.get_appointment_by_id(db, appointment_id, user.current_tenant_id)
    if not appt:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Appointment not found")
    full = calendar_service.to_appointment_out(db, user.current_tenant_id, appt)
    return create_success_response(
        data=AppointmentDetailOut.model_validate(
            full.model_dump(exclude={"slot_start", "slot_end"})
        )
    )


@router.patch(
    "/appointments/{appointment_id}/reschedule",
    response_model=SuccessResponse[AppointmentOut],
)
def reschedule_appointment(
    appointment_id: uuid.UUID,
    payload: AppointmentReschedule,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Move a confirmed/pending appointment's local record to a new time."""
    try:
        appt = calendar_service.reschedule_appointment(
            db=db,
            tenant_id=user.current_tenant_id,
            appointment_id=appointment_id,
            slot_start=payload.slot_start,
            duration_minutes=payload.duration_minutes,
            customer_name=payload.customer_name,
            customer_phone=payload.customer_phone,
            customer_email=payload.customer_email,
            appointment_reason=payload.appointment_reason,
            notes=payload.notes,
            notify_user_id=user.id,
        )
    except ValueError as exc:
        msg = str(exc)
        if msg == "Appointment not found.":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=msg)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=msg)
    return create_success_response(
        data=calendar_service.to_appointment_out(db, user.current_tenant_id, appt)
    )


@router.patch("/appointments/{appointment_id}", response_model=SuccessResponse[AppointmentOut])
def update_appointment_status(
    appointment_id: uuid.UUID,
    payload: AppointmentStatusUpdate,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Update appointment status (confirm/cancel/complete/no_show)."""
    try:
        appt = calendar_service.update_appointment_status(
            db=db,
            appointment_id=appointment_id,
            tenant_id=user.current_tenant_id,
            status=payload.status,
            cancellation_reason=payload.cancellation_reason,
            notes=payload.notes,
            notify_user_id=user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    if not appt:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Appointment not found")
    return create_success_response(
        data=calendar_service.to_appointment_out(db, user.current_tenant_id, appt)
    )


@router.delete("/appointments/{appointment_id}", response_model=SuccessResponse[dict])
def delete_appointment(
    appointment_id: uuid.UUID,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Permanently delete an appointment."""
    deleted = calendar_service.delete_appointment(db, appointment_id, user.current_tenant_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Appointment not found")
    return create_success_response(data={"deleted": True, "id": str(appointment_id)})


# ─── Business Hours (used by the Smart Callback Scheduler retry gate) ─────────

@router.get("/business-hours", response_model=SuccessResponse[List[BusinessHoursOut]])
def get_business_hours(
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    hours = business_hours_service.get_business_hours(db, user.current_tenant_id)
    out = [BusinessHoursOut.model_validate(h) for h in hours]
    return create_success_response(data=_map_hours_out_to_api(out))


@router.post("/business-hours", response_model=SuccessResponse[List[BusinessHoursOut]], status_code=status.HTTP_201_CREATED)
def create_business_hours(
    payload: List[BusinessHoursUpsert],
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Create business hours for the tenant. Use PUT to update existing weekdays."""
    internal_payload = _map_hours_payload_to_internal(payload)
    try:
        hours = business_hours_service.create_business_hours(db, user.current_tenant_id, internal_payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except BusinessHoursConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "Business hours already exist for one or more weekdays.",
                "day_of_week": [_internal_day_to_api(day) for day in exc.days],
            },
        ) from exc
    out = [BusinessHoursOut.model_validate(h) for h in hours]
    return create_success_response(data=_map_hours_out_to_api(out))


@router.put("/business-hours", response_model=SuccessResponse[List[BusinessHoursOut]])
def upsert_business_hours(
    payload: List[BusinessHoursUpsert],
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Set business hours for the tenant. Pass all 7 days at once (or just the ones you want to update)."""
    internal_payload = _map_hours_payload_to_internal(payload)
    hours = business_hours_service.upsert_business_hours(db, user.current_tenant_id, internal_payload)
    out = [BusinessHoursOut.model_validate(h) for h in hours]
    return create_success_response(data=_map_hours_out_to_api(out))


@router.delete("/business-hours/{business_hours_id}", response_model=SuccessResponse[dict])
def delete_business_hours(
    business_hours_id: uuid.UUID,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    deleted = business_hours_service.delete_business_hours(db, business_hours_id, user.current_tenant_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Business hours not found")
    return create_success_response(data={"deleted": True, "id": str(business_hours_id)})
