from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from datetime import time
import uuid


# ─── Business Hours ────────────────────────────────────────────────────────────
# Kept for the Smart Callback Scheduler (app/services/callback_scheduler_service.py),
# which reads business hours to gate retry timing. Unrelated to appointment-slot
# availability, which now lives in Calendly.

class BusinessHoursUpsert(BaseModel):
    day_of_week: int = Field(..., ge=0, le=6, description="0=Sunday … 6=Saturday")
    open_time: str | None = Field(None, description="HH:MM, e.g. '09:00'")
    close_time: str | None = Field(None, description="HH:MM, e.g. '17:00'")
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
    day_of_week: int = Field(..., description="0=Sunday … 6=Saturday")
    open_time: str | None = None
    close_time: str | None = None
    is_closed: bool
    timezone: str
    slot_duration_minutes: int

    @field_validator("open_time", "close_time", mode="before")
    @classmethod
    def _coerce_time_to_str(cls, v):
        if isinstance(v, time):
            return v.strftime("%H:%M")
        return v
