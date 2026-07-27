from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


_SPOKEN_HINT = (
    "Store in natural spoken form — the agent will read this aloud during calls. "
    "Avoid symbols, abbreviations, or special characters. "
    "Example phone: 'one eight hundred five five five one two three four'. "
    "Example website: 'our website is tee gee ess dot com'."
)


class BusinessKnowledgeCreate(BaseModel):
    label: str = Field(..., max_length=255, description="Admin label to identify this record")
    agent_id: uuid.UUID | None = Field(
        default=None,
        description="If set, this record is scoped to a specific agent. "
                    "Otherwise it applies tenant-wide.",
    )

    business_name: str | None = Field(default=None, max_length=255)
    business_type: str | None = Field(default=None, max_length=255)
    business_description: str | None = Field(default=None, description=_SPOKEN_HINT)

    address: str | None = Field(default=None, description=_SPOKEN_HINT)
    phone: str | None = Field(
        default=None,
        max_length=255,
        description=_SPOKEN_HINT,
    )
    email: str | None = Field(default=None, max_length=255, description=_SPOKEN_HINT)
    website_url: str | None = Field(default=None, max_length=512, description=_SPOKEN_HINT)

    primary_service: str | None = Field(default=None)
    secondary_service: str | None = Field(default=None)
    service_areas: str | None = Field(default=None)
    specializations: str | None = Field(default=None)

    pricing_information: str | None = Field(default=None)
    additional_information: str | None = Field(default=None)


class BusinessKnowledgeUpdate(BaseModel):
    label: str | None = Field(default=None, max_length=255)
    agent_id: uuid.UUID | None = None

    business_name: str | None = Field(default=None, max_length=255)
    business_type: str | None = Field(default=None, max_length=255)
    business_description: str | None = None

    address: str | None = None
    phone: str | None = Field(default=None, max_length=255)
    email: str | None = Field(default=None, max_length=255)
    website_url: str | None = Field(default=None, max_length=512)

    primary_service: str | None = None
    secondary_service: str | None = None
    service_areas: str | None = None
    specializations: str | None = None

    pricing_information: str | None = None
    additional_information: str | None = None


class BusinessKnowledgeOut(BaseModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    agent_id: uuid.UUID | None = None

    label: str

    business_name: str | None = None
    business_type: str | None = None
    business_description: str | None = None

    address: str | None = None
    phone: str | None = None
    email: str | None = None
    website_url: str | None = None

    primary_service: str | None = None
    secondary_service: str | None = None
    service_areas: str | None = None
    specializations: str | None = None

    pricing_information: str | None = None
    additional_information: str | None = None

    is_active: bool
    created_at: datetime
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class BusinessKnowledgeList(BaseModel):
    items: List[BusinessKnowledgeOut]
    total: int
