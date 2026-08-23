"""Workspace (tenant) request/response schemas — Pydantic v2."""
from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class WorkspaceCreate(BaseModel):
    """Request body for ``POST /api/v1/workspace``."""

    name: str = Field(..., min_length=3, max_length=50, examples=["Acme Corp"])

    model_config = ConfigDict(extra="forbid")

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, v: str) -> str:
        normalized = " ".join(v.split())
        if not 3 <= len(normalized) <= 50:
            raise ValueError("name must be 3-50 characters after trimming whitespace")
        return normalized


class WorkspaceUpdateName(WorkspaceCreate):
    """Request body for ``PUT /api/v1/workspace/name``."""


class _WorkspaceBase(BaseModel):
    id: uuid.UUID = Field(examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"])
    name: str = Field(examples=["Acme Corp"])
    created_at: datetime = Field(serialization_alias="createdAt", examples=["2025-01-15T10:30:00Z"])

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class WorkspaceCreatedOut(_WorkspaceBase):
    """Minimal response shape per ticket: ``{id, name, createdAt}``."""


class WorkspaceOut(_WorkspaceBase):
    """Full (non-internal) workspace projection. Hides ``deleted_at``, ``schema_name`` etc."""

    status: str
    credits: float

    @field_validator("credits", mode="before")
    @classmethod
    def _coerce_credits(cls, v: Any) -> float:
        if v is None:
            return 0.0
        if isinstance(v, Decimal):
            return round(float(v), 2)
        return round(float(v), 2)


class BrandingConfigUpsert(BaseModel):
    """Request body for PUT /api/v2/workspace/branding"""
    logo_url: Any | None = None  # Will be validated as HttpUrl below
    primary_colour: str = Field(..., pattern=r"^#[0-9A-Fa-f]{6}$")
    display_name: str

    @field_validator("logo_url")
    @classmethod
    def _validate_logo_url(cls, v: Any) -> Any:
        if v is None:
            return v
        from pydantic_core import Url
        if isinstance(v, str):
            if not v.startswith("https://"):
                raise ValueError("logo_url must be an HTTPS URL")
            return v
        if isinstance(v, Url):
            if v.scheme != "https":
                raise ValueError("logo_url must be an HTTPS URL")
            return str(v)
        return v

class BrandingConfigOut(BaseModel):
    """Response shape for branding config"""
    logo_url: str | None = None
    primary_colour: str | None = None
    display_name: str | None = None

    model_config = ConfigDict(from_attributes=True)


class PricingConfigUpsert(BaseModel):
    """Request body for PUT /api/v2/workspace/pricing"""
    per_minute_rate: Decimal
    markup_percent: Decimal = Field(..., ge=0, le=500)


class SurchargeInfoOut(BaseModel):
    """A single named per-minute surcharge that can stack on top of the base
    per-minute rate (see app.services.credit_service.SURCHARGE_CATALOG).

    This is the full catalog of surcharges the platform knows how to apply —
    not necessarily all active for every agent/call — since these endpoints
    are workspace-scoped rather than agent-scoped. `applies_when` documents
    the condition that activates it for a given call.
    """
    key: str
    label: str
    rate_per_minute: Decimal
    applies_when: str

    model_config = ConfigDict(from_attributes=True)


_SURCHARGE_FIELD_DESCRIPTION = (
    "Full catalog of per-minute surcharges the platform knows how to apply "
    "(e.g. OpenAI Realtime, ElevenLabs voice) — this is NOT the list of "
    "surcharges currently being charged on any specific call or agent. "
    "Whether a given surcharge is actually active depends on that agent's "
    "model/TTS configuration at call time; these endpoints are "
    "workspace-scoped, not agent- or call-scoped, so this field only "
    "advertises what CAN stack on top of the base per-minute rate."
)


class PricingConfigOut(BaseModel):
    """Response shape for pricing config"""
    per_minute_rate: Decimal
    markup_percent: Decimal
    effective_client_rate: Decimal
    available_surcharges: list[SurchargeInfoOut] = Field(
        default_factory=list, description=_SURCHARGE_FIELD_DESCRIPTION
    )

    model_config = ConfigDict(from_attributes=True)


class WorkspaceUsageOut(BaseModel):
    """Response shape for cycle usage"""
    minutes_used_this_cycle: Decimal
    minutes_included: Decimal | None = None
    overage_minutes: Decimal
    overage_cost: Decimal
    available_surcharges: list[SurchargeInfoOut] = Field(
        default_factory=list, description=_SURCHARGE_FIELD_DESCRIPTION
    )

class SubAccountCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=50)
    contact_email: EmailStr

class SubAccountUpdate(BaseModel):
    name: str | None = Field(None, min_length=3, max_length=50)
    contact_email: EmailStr | None = None

class SubAccountOut(BaseModel):
    id: uuid.UUID
    name: str
    contact_email: str | None = None
    status: str
    api_key_prefix: str | None = None
    usage_this_cycle_minutes: float

    model_config = {
        "from_attributes": True
    }

class SubAccountCreateOut(SubAccountOut):
    api_key: str

class SubAccountListOut(BaseModel):
    data: list[SubAccountOut]
    total: int
    page: int
    page_size: int


class WorkspaceBreakdownRowOut(BaseModel):
    """One row of the agency billing-dashboard breakdown table — either a
    real workspace (the "Master" parent tenant or one of its direct
    sub-accounts) or the synthetic "Total" summary row."""

    workspace_id: uuid.UUID | None = None
    name: str
    is_master: bool = False
    this_month: Decimal
    all_time: Decimal
    avg_monthly: Decimal
    growth_percent: float | None = Field(
        default=None,
        description=(
            "% change of this-month-to-date spend vs. last full calendar "
            "month. None when last month had $0 spend (avoids divide-by-zero "
            "— the mockup renders this as '—')."
        ),
    )

    model_config = ConfigDict(from_attributes=True)


class WorkspaceUsageBreakdownOut(BaseModel):
    """Response for GET /api/v2/workspace/usage/breakdown."""

    period: str = Field(
        default="this_month",
        description="Reserved for a future period selector; only 'this_month' is supported today.",
    )
    this_month_total: Decimal
    this_month_growth_percent: float | None = None
    all_time_total: Decimal
    avg_per_workspace_this_month: Decimal
    workspace_count: int
    active_workspace_count_this_month: int
    rows: list[WorkspaceBreakdownRowOut]
    totals: WorkspaceBreakdownRowOut


class RecentActivityItemOut(BaseModel):
    """One row of the "Recent Activity" table."""

    date: datetime
    type: str = "call_usage"
    description: str
    amount: Decimal = Field(description="Negative — credits deducted for this usage row.")

    model_config = ConfigDict(from_attributes=True)


class RecentActivityOut(BaseModel):
    items: list[RecentActivityItemOut]


class MonthlyMinutesUsageOut(BaseModel):
    """One row of the "Minutes Usage" table — one calendar month."""

    month: str = Field(description="ISO calendar-month key, e.g. '2026-08'.")
    label: str = Field(description="Human-readable label, e.g. 'August 2026'.")
    total_minutes: Decimal
    call_count: int
    total_cost: Decimal


class MinutesByMonthOut(BaseModel):
    months: list[MonthlyMinutesUsageOut]


class MemberRoleUpdate(BaseModel):
    """Request body for PUT /api/v2/workspace/members/{user_id}/role"""
    role: str


class MemberRoleOut(BaseModel):
    """Response shape after a member's role is updated"""
    user_id: uuid.UUID
    workspace_id: uuid.UUID
    role: str
