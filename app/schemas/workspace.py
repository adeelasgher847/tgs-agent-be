"""Workspace (tenant) request/response schemas — Pydantic v2."""
from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
