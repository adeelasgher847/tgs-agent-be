"""Response schemas for the dashboard home-view summary endpoint."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import List

from pydantic import BaseModel, ConfigDict

from app.schemas.call_flow import CallFlowListItem


class DashboardKbItem(BaseModel):
    """Minimal knowledge-base summary card for the dashboard."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None
    file_count: int
    created_at: datetime


class DashboardRecentCallItem(BaseModel):
    """Minimal call summary card for the dashboard."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: str
    summary: str | None
    call_type: str | None
    duration: int | None
    cost: float | None
    created_at: datetime


class DashboardSummary(BaseModel):
    """All data needed to render the dashboard home view in a single response."""

    monthly_spent: int
    monthly_calls: int
    success_rate_percent: int | None
    credits: float
    recent_call_flows: List[CallFlowListItem]
    recent_knowledge_bases: List[DashboardKbItem]
    recent_calls: List[DashboardRecentCallItem]
