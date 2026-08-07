from __future__ import annotations

import uuid
from datetime import datetime
from typing import List

from pydantic import BaseModel, ConfigDict


class CallHistoryMetrics(BaseModel):
    total_calls: int
    completed: int
    failed: int
    no_answer: int
    avg_duration_seconds: float | None
    total_duration_seconds: int | None
    success_rate_percent: float | None
    total_cost: float | None = None


class CallHistoryTimeSeriesPoint(BaseModel):
    date: str  # YYYY-MM-DD
    total: int
    completed: int
    failed: int


class CallHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    call_id: uuid.UUID
    direction: str
    from_number: str | None
    to_number: str | None
    agent_name: str | None
    flow_name: str | None
    status: str
    duration_seconds: int | None
    started_at: datetime | None
    ended_at: datetime | None
    ab_variant: str | None = None


class CallHistoryList(BaseModel):
    items: List[CallHistoryItem]
    total: int
    page: int
    per_page: int
    pages: int


class BatchCallMetrics(BaseModel):
    total_batches: int
    avg_completion_rate_percent: float | None
    total_calls_dispatched: int
    total_failed: int
