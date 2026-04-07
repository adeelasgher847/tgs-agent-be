from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Optional, List
from datetime import datetime, date, time
import uuid


# ─── Business Hours ───────────────────────────────────────────────────────────

class BusinessHoursUpsert(BaseModel):
    day_of_week: int = Field(..., ge=0, le=6, description="0=Monday … 6=Sunday")
    open_time: Optional[str] = Field(None, description="HH:MM, e.g. '09:00'")
    close_time: Optional[str] = Field(None, description="HH:MM, e.g. '17:00'")
    is_closed: bool = False
    timezone: str = Field(default="UTC", description="IANA timezone, e.g. 'Asia/Karachi'")
    slot_duration_minutes: int = Field(default=30, ge=15, le=120)

    @model_validator(mode="after")
    def validate_hours(self):
        if self.is_closed:
            return self

        if not self.open_time or not self.close_time:
            raise ValueError("Open and close times are required when the business is open.")

        try:
            open_value = time.fromisoformat(self.open_time)
            close_value = time.fromisoformat(self.close_time)
        except ValueError as exc:
            raise ValueError("Business hours must use HH:MM format.") from exc

        if open_value >= close_value:
            raise ValueError("Close time must be after open time.")

        return self


class BusinessHoursOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    tenant_id: uuid.UUID
    day_of_week: int
    open_time: Optional[str] = None
    close_time: Optional[str] = None
    is_closed: bool
    timezone: str
    slot_duration_minutes: int

    @field_validator("open_time", "close_time", mode="before")
    @classmethod
    def _coerce_time_to_str(cls, v):
        if isinstance(v, time):
            return v.strftime("%H:%M")
        return v

# ─── Blocked Slots ────────────────────────────────────────────────────────────

class BlockedSlotCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    blocked_from: datetime
    blocked_until: datetime

    @model_validator(mode="after")
    def validate_range(self):
        if self.blocked_until <= self.blocked_from:
            raise ValueError("blocked_until must be after blocked_from.")
        return self


class BlockedSlotOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    tenant_id: uuid.UUID
    title: str
    blocked_from: datetime
    blocked_until: datetime
    created_at: datetime

# ─── Appointments ─────────────────────────────────────────────────────────────

class AppointmentCreate(BaseModel):
    customer_name: str = Field(..., min_length=1, max_length=255)
    customer_phone: str = Field(..., min_length=5, max_length=50)
    customer_email: Optional[str] = None
    appointment_reason: Optional[str] = None
    slot_start: datetime
    agent_id: Optional[uuid.UUID] = None
    notes: Optional[str] = None


class AppointmentStatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(pending|confirmed|cancelled|completed|no_show)$")
    cancellation_reason: Optional[str] = None
    notes: Optional[str] = None


class AppointmentReschedule(BaseModel):
    """Move appointment to a new slot. Optional fields update customer details when provided."""

    slot_start: datetime
    duration_minutes: Optional[int] = Field(None, ge=15, le=120)
    customer_name: Optional[str] = Field(None, min_length=1, max_length=255)
    customer_phone: Optional[str] = Field(None, min_length=5, max_length=50)
    customer_email: Optional[str] = None
    appointment_reason: Optional[str] = None
    notes: Optional[str] = None


class AppointmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    tenant_id: uuid.UUID
    agent_id: Optional[uuid.UUID] = None
    customer_name: str
    customer_phone: str
    customer_email: Optional[str] = None
    appointment_reason: Optional[str] = None
    slot_start: datetime
    slot_end: datetime
    duration_minutes: int
    status: str
    created_via: str
    notes: Optional[str] = None
    cancellation_reason: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    # Additive: same instants as slot_start/slot_end, expressed in business-hours timezone
    business_timezone: Optional[str] = Field(
        None,
        description="IANA timezone from business hours (for slot_*_local). slot_start/slot_end remain UTC.",
    )
    slot_start_local: Optional[datetime] = Field(
        None,
        description="slot_start converted to business_timezone (same instant as slot_start).",
    )
    slot_end_local: Optional[datetime] = Field(
        None,
        description="slot_end converted to business_timezone (same instant as slot_end).",
    )


class AppointmentListItemOut(BaseModel):
    id: uuid.UUID
    appointment_reason: Optional[str] = None
    slot_start_local: Optional[datetime] = None
    slot_end_local: Optional[datetime] = None


class AppointmentListResponse(BaseModel):
    appointments: List[AppointmentListItemOut]
    total: int


# ─── Slot availability ────────────────────────────────────────────────────────

class AvailableSlot(BaseModel):
    slot_start: datetime
    slot_end: datetime
    slot_label: str   # "9:00 AM", "10:30 AM"


class AvailableSlotsResponse(BaseModel):
    date: str
    timezone: str
    slots: List[AvailableSlot]
    total: int
