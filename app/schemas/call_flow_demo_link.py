from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class CallFlowDemoLinkCreate(BaseModel):
    """Request body for ``POST /api/v1/call-flows/{flow_id}/demo-links``."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str | None = Field(None, max_length=255)
    expires_at: datetime | None = Field(None, alias="expiresAt")
    per_user_limit_minutes: Decimal | None = Field(None, alias="perUserLimitMinutes", gt=0)
    total_budget_minutes: Decimal | None = Field(None, alias="totalBudgetMinutes", gt=0)


class CallFlowDemoLinkUpdate(BaseModel):
    """Request body for ``PATCH /api/v1/call-flows/demo-links/{link_id}``."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str | None = Field(None, max_length=255)
    expires_at: datetime | None = Field(None, alias="expiresAt")
    per_user_limit_minutes: Decimal | None = Field(None, alias="perUserLimitMinutes", gt=0)
    total_budget_minutes: Decimal | None = Field(None, alias="totalBudgetMinutes", gt=0)
    is_active: bool | None = Field(None, alias="isActive")


class CallFlowDemoLinkOut(BaseModel):
    """Response shape for a demo link, including the derived shareable URL."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: uuid.UUID
    call_flow_id: uuid.UUID = Field(..., serialization_alias="callFlowId")
    name: str | None = None
    token: str
    share_url: str | None = Field(None, serialization_alias="shareUrl")
    is_active: bool = Field(..., serialization_alias="isActive")
    expires_at: datetime | None = Field(None, serialization_alias="expiresAt")
    per_user_limit_minutes: Decimal | None = Field(None, serialization_alias="perUserLimitMinutes")
    total_budget_minutes: Decimal | None = Field(None, serialization_alias="totalBudgetMinutes")
    total_minutes_used: Decimal = Field(..., serialization_alias="totalMinutesUsed")
    created_by_user_id: uuid.UUID | None = Field(None, serialization_alias="createdByUserId")
    created_at: datetime = Field(..., serialization_alias="createdAt")
    updated_at: datetime | None = Field(None, serialization_alias="updatedAt")


class CallFlowDemoLinkListResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    data: list[CallFlowDemoLinkOut]
