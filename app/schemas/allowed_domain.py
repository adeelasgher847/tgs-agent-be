"""Allowed-domain (Web SDK whitelist) request/response schemas — Pydantic v2."""
from __future__ import annotations

import uuid
from datetime import datetime
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AllowedDomainCreate(BaseModel):
    """Request body for ``POST /api/v1/workspace/allowed-domains``."""

    model_config = ConfigDict(extra="forbid")

    domain: str = Field(..., min_length=1, max_length=255, examples=["https://app.example.com"])

    @field_validator("domain")
    @classmethod
    def _validate_https_url(cls, v: str) -> str:
        parsed = urlsplit(v.strip())
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError(
                "domain must be a valid HTTPS URL, e.g. https://app.example.com"
            )

        from app.core.origin import normalize_origin

        return normalize_origin(v)


class AllowedDomainOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: uuid.UUID
    domain: str
    created_at: datetime = Field(serialization_alias="createdAt")
