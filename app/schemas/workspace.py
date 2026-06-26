"""Workspace (tenant) request/response schemas — Pydantic v2."""
from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

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
            return float(v)
        return float(v)


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
        from pydantic import HttpUrl
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


class PricingConfigOut(BaseModel):
    """Response shape for pricing config"""
    per_minute_rate: Decimal
    markup_percent: Decimal
    effective_client_rate: Decimal

    model_config = ConfigDict(from_attributes=True)


class WorkspaceUsageOut(BaseModel):
    """Response shape for cycle usage"""
    minutes_used_this_cycle: Decimal
    minutes_included: Optional[Decimal] = None
    overage_minutes: Decimal
    overage_cost: Decimal

class SubAccountCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=50)
    contact_email: EmailStr

class SubAccountUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=3, max_length=50)
    contact_email: Optional[EmailStr] = None

class SubAccountOut(BaseModel):
    id: uuid.UUID
    name: str
    contact_email: Optional[str] = None
    status: str
    api_key_prefix: Optional[str] = None
    usage_this_cycle_minutes: float

    class Config:
        orm_mode = True

class SubAccountCreateOut(SubAccountOut):
    api_key: str

class SubAccountListOut(BaseModel):
    data: list[SubAccountOut]
    total: int
    page: int
    page_size: int


class MemberRoleUpdate(BaseModel):
    """Request body for PUT /api/v2/workspace/members/{user_id}/role"""
    role: str


class MemberRoleOut(BaseModel):
    """Response shape after a member's role is updated"""
    user_id: uuid.UUID
    workspace_id: uuid.UUID
    role: str
