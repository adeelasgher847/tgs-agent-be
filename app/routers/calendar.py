from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import date
import uuid

from app.api.deps import get_db, require_tenant
from app.models.user import User
from app.schemas.base import SuccessResponse
from app.schemas.calendar import (
    BusinessHoursUpsert, BusinessHoursOut,
    BlockedSlotCreate, BlockedSlotOut,
    AppointmentCreate, AppointmentStatusUpdate, AppointmentReschedule, AppointmentOut,
    AppointmentListResponse,
    AvailableSlotsResponse,
)
from app.services.calendar_service import BusinessHoursConflictError, calendar_service
from app.utils.response import create_success_response

router = APIRouter()


# ─── Slot Availability ────────────────────────────────────────────────────────

@router.get("/slots", response_model=SuccessResponse[AvailableSlotsResponse])
def get_available_slots(
    date: date = Query(..., description="Date to check (YYYY-MM-DD)"),
    agent_id: Optional[uuid.UUID] = Query(None),
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Return available booking slots for a given date."""
    result = calendar_service.get_available_slots(
        db=db,
        tenant_id=user.current_tenant_id,
        target_date=date,
        agent_id=agent_id,
    )
    return create_success_response(data=result)


# ─── Appointments ─────────────────────────────────────────────────────────────

@router.post("/appointments", response_model=SuccessResponse[AppointmentOut], status_code=status.HTTP_201_CREATED)
def create_appointment(
    payload: AppointmentCreate,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Book an appointment. Returns 409 if the slot is unavailable."""
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
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return create_success_response(
        data=calendar_service.to_appointment_out(db, user.current_tenant_id, appt),
        status_code=status.HTTP_201_CREATED,
    )


@router.get("/appointments", response_model=SuccessResponse[AppointmentListResponse])
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
                calendar_service.to_appointment_out(db, user.current_tenant_id, a) for a in items
            ],
            total=total,
        )
    )


@router.get("/appointments/{appointment_id}", response_model=SuccessResponse[AppointmentOut])
def get_appointment(
    appointment_id: uuid.UUID,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    appt = calendar_service.get_appointment_by_id(db, appointment_id, user.current_tenant_id)
    if not appt:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Appointment not found")
    return create_success_response(
        data=calendar_service.to_appointment_out(db, user.current_tenant_id, appt)
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
    """Move a confirmed/pending appointment to a new time. Fails with 409 if the slot is unavailable."""
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


# ─── Business Hours ───────────────────────────────────────────────────────────

@router.get("/business-hours", response_model=SuccessResponse[List[BusinessHoursOut]])
def get_business_hours(
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    hours = calendar_service.get_business_hours(db, user.current_tenant_id)
    return create_success_response(data=[BusinessHoursOut.model_validate(h) for h in hours])


@router.post("/business-hours", response_model=SuccessResponse[List[BusinessHoursOut]], status_code=status.HTTP_201_CREATED)
def create_business_hours(
    payload: List[BusinessHoursUpsert],
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Create business hours for the tenant. Use PUT to update existing weekdays."""
    try:
        hours = calendar_service.create_business_hours(db, user.current_tenant_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except BusinessHoursConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "Business hours already exist for one or more weekdays.",
                "day_of_week": exc.days,
            },
        ) from exc
    return create_success_response(data=[BusinessHoursOut.model_validate(h) for h in hours])


@router.put("/business-hours", response_model=SuccessResponse[List[BusinessHoursOut]])
def upsert_business_hours(
    payload: List[BusinessHoursUpsert],
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Set business hours for the tenant. Pass all 7 days at once (or just the ones you want to update)."""
    hours = calendar_service.upsert_business_hours(db, user.current_tenant_id, payload)
    return create_success_response(data=[BusinessHoursOut.model_validate(h) for h in hours])


@router.delete("/business-hours/{business_hours_id}", response_model=SuccessResponse[dict])
def delete_business_hours(
    business_hours_id: uuid.UUID,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    deleted = calendar_service.delete_business_hours(db, business_hours_id, user.current_tenant_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Business hours not found")
    return create_success_response(data={"deleted": True, "id": str(business_hours_id)})


# ─── Blocked Slots ────────────────────────────────────────────────────────────

@router.get("/blocked-slots", response_model=SuccessResponse[List[BlockedSlotOut]])
def list_blocked_slots(
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """List all blocked slots for the tenant."""
    slots = calendar_service.get_blocked_slots(db, user.current_tenant_id)
    return create_success_response(data=[BlockedSlotOut.model_validate(s) for s in slots])


@router.post("/blocked-slots", response_model=SuccessResponse[BlockedSlotOut], status_code=status.HTTP_201_CREATED)
def create_blocked_slot(
    payload: BlockedSlotCreate,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    bs = calendar_service.create_blocked_slot(db, user.current_tenant_id, payload)
    return create_success_response(data=BlockedSlotOut.model_validate(bs), status_code=status.HTTP_201_CREATED)


@router.delete("/blocked-slots/{blocked_slot_id}", response_model=SuccessResponse[dict])
def delete_blocked_slot(
    blocked_slot_id: uuid.UUID,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    deleted = calendar_service.delete_blocked_slot(db, blocked_slot_id, user.current_tenant_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Blocked slot not found")
    return create_success_response(data={"deleted": True, "id": str(blocked_slot_id)})
