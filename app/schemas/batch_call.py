from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Request schemas ──────────────────────────────────────────────────────────

class BatchJobCreate(BaseModel):
    """Parsed from multipart/form-data after CSV is validated."""

    agent_id: uuid.UUID
    scheduled_at: Optional[datetime] = None


# ── Response schemas ─────────────────────────────────────────────────────────

class BatchCallRecordOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    batch_job_id: uuid.UUID
    phone_number: str
    variables: Optional[Dict[str, Any]] = None
    status: str
    call_id: Optional[uuid.UUID] = None
    attempts: int
    last_error: Optional[str] = None
    next_attempt_at: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


class BatchJobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    workspace_id: uuid.UUID
    agent_id: Optional[uuid.UUID] = None
    status: str
    total_count: int
    waiting_count: int
    active_count: int
    completed_count: int
    failed_count: int
    gcs_path: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime


class BatchJobProgress(BaseModel):
    """Live progress snapshot for GET /batch-calls/{batch_id}/progress."""

    batch_id: uuid.UUID
    status: str
    waiting: int
    active: int
    completed: int
    failed: int
    total: int
    percent_complete: float


class PaginatedBatchJobs(BaseModel):
    items: List[BatchJobOut]
    total: int
    page: int
    page_size: int


class PaginatedBatchCallRecords(BaseModel):
    items: List[BatchCallRecordOut]
    total: int
    page: int
    page_size: int


# ── Validation error detail ───────────────────────────────────────────────────

class CsvRowError(BaseModel):
    row: int
    error: str


class CsvValidationError(BaseModel):
    message: str
    errors: List[CsvRowError] = Field(default_factory=list)
