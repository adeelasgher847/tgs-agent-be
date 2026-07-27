from datetime import datetime
from typing import Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field


class ApiKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)


class ApiKeyOut(BaseModel):
    """API key as returned from list/get — never includes the raw secret."""

    id: uuid.UUID
    name: str
    workspace_id: uuid.UUID
    masked_key: str
    is_active: bool
    created_at: datetime
    last_used_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class ApiKeyCreated(ApiKeyOut):
    """Create response — raw key is only present in this one-time payload."""

    raw_key: str
