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
    agent_id: Optional[uuid.UUID] = Field(
        default=None,
        description="If set, this record is scoped to a specific agent. "
                    "Otherwise it applies tenant-wide.",
    )

    business_name: Optional[str] = Field(default=None, max_length=255)
    business_type: Optional[str] = Field(default=None, max_length=255)
    business_description: Optional[str] = Field(default=None, description=_SPOKEN_HINT)

    address: Optional[str] = Field(default=None, description=_SPOKEN_HINT)
    phone: Optional[str] = Field(
        default=None,
        max_length=255,
        description=_SPOKEN_HINT,
    )
    email: Optional[str] = Field(default=None, max_length=255, description=_SPOKEN_HINT)
    website_url: Optional[str] = Field(default=None, max_length=512, description=_SPOKEN_HINT)

    primary_service: Optional[str] = Field(default=None)
    secondary_service: Optional[str] = Field(default=None)
    service_areas: Optional[str] = Field(default=None)
    specializations: Optional[str] = Field(default=None)

    pricing_information: Optional[str] = Field(default=None)
    additional_information: Optional[str] = Field(default=None)


class BusinessKnowledgeUpdate(BaseModel):
    label: Optional[str] = Field(default=None, max_length=255)
    agent_id: Optional[uuid.UUID] = None

    business_name: Optional[str] = Field(default=None, max_length=255)
    business_type: Optional[str] = Field(default=None, max_length=255)
    business_description: Optional[str] = None

    address: Optional[str] = None
    phone: Optional[str] = Field(default=None, max_length=255)
    email: Optional[str] = Field(default=None, max_length=255)
    website_url: Optional[str] = Field(default=None, max_length=512)

    primary_service: Optional[str] = None
    secondary_service: Optional[str] = None
    service_areas: Optional[str] = None
    specializations: Optional[str] = None

    pricing_information: Optional[str] = None
    additional_information: Optional[str] = None


class BusinessKnowledgeOut(BaseModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    agent_id: Optional[uuid.UUID] = None

    label: str

    business_name: Optional[str] = None
    business_type: Optional[str] = None
    business_description: Optional[str] = None

    address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    website_url: Optional[str] = None

    primary_service: Optional[str] = None
    secondary_service: Optional[str] = None
    service_areas: Optional[str] = None
    specializations: Optional[str] = None

    pricing_information: Optional[str] = None
    additional_information: Optional[str] = None

    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class BusinessKnowledgeList(BaseModel):
    items: List[BusinessKnowledgeOut]
    total: int
