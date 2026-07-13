from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class CallHistoryMetrics(BaseModel):
    total_calls: int
    completed: int
    failed: int
    no_answer: int
    avg_duration_seconds: Optional[float]
    total_duration_seconds: Optional[int]
    success_rate_percent: Optional[float]


class CallHistoryTimeSeriesPoint(BaseModel):
    date: str  # YYYY-MM-DD
    total: int
    completed: int
    failed: int


class CallHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    call_id: uuid.UUID
    direction: str
    from_number: Optional[str]
    to_number: Optional[str]
    agent_name: Optional[str]
    flow_name: Optional[str]
    status: str
    duration_seconds: Optional[int]
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    ab_variant: Optional[str] = None


class CallHistoryList(BaseModel):
    items: List[CallHistoryItem]
    total: int
    page: int
    per_page: int
    pages: int


class BatchCallMetrics(BaseModel):
    total_batches: int
    avg_completion_rate_percent: Optional[float]
    total_calls_dispatched: int
    total_failed: int
