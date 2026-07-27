from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from zoneinfo import available_timezones

from pydantic import BaseModel, Field, field_validator, model_validator


class GapInterval(BaseModel):
    """A single time gap between callback attempts."""

    days: int = Field(default=0, ge=0, description="Number of days to wait")
    hours: int = Field(default=0, ge=0, description="Number of hours to wait")

    @model_validator(mode="after")
    def at_least_one_unit(self) -> "GapInterval":
        if self.days == 0 and self.hours == 0:
            raise ValueError("Each gap interval must have days > 0 or hours > 0")
        return self


class CallbackConfigUpdate(BaseModel):
    """
    PUT /api/v1/agents/{id}/callback-config request body.
    Validates the timezone against the IANA zoneinfo database.
    """

    smart_callback_enabled: bool
    max_attempts: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum retry attempts before the chain is exhausted",
    )
    gap_schedule: List[GapInterval] = Field(
        default_factory=list,
        description="Ordered list of gaps between attempts; index 0 = gap after first miss",
    )
    timezone: str = Field(description="IANA timezone, e.g. 'America/New_York'")

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        if v not in available_timezones():
            raise ValueError(
                f"'{v}' is not a valid IANA timezone. "
                "Use a value from zoneinfo.available_timezones()."
            )
        return v

    @model_validator(mode="after")
    def gap_schedule_length(self) -> "CallbackConfigUpdate":
        """
        When enabled, gap_schedule must have at least one entry and at most
        max_attempts entries (one gap per inter-attempt interval).
        """
        if self.smart_callback_enabled and not self.gap_schedule:
            raise ValueError(
                "gap_schedule must contain at least one interval when smart_callback_enabled is True"
            )
        if len(self.gap_schedule) > self.max_attempts:
            raise ValueError(
                f"gap_schedule length ({len(self.gap_schedule)}) cannot exceed "
                f"max_attempts ({self.max_attempts})"
            )
        return self


class CallbackConfigResponse(BaseModel):
    """Response body for PUT /api/v1/agents/{id}/callback-config."""

    smart_callback_enabled: bool
    max_attempts: int
    gap_schedule: List[GapInterval]
    timezone: Optional[str]


class CallbackStatusResponse(BaseModel):
    """
    GET /api/v1/agents/{id}/callback-status response.
    Returns live counters alongside the static config.
    """

    enabled: bool
    max_attempts: int
    gap_schedule: List[GapInterval]
    timezone: Optional[str]
    pending_retries: int = Field(description="Number of pending callbacks for this agent")
    next_scheduled_at: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp of the earliest pending callback",
    )


class CallbackHistoryItem(BaseModel):
    """One row from GET /api/v1/calls/{call_id}/callback-history."""

    attempt_number: int
    scheduled_at: datetime
    executed_at: Optional[datetime]
    status: str
