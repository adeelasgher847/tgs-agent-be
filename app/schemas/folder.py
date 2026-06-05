from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class FolderCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(..., min_length=1, max_length=255)


class FolderUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(..., min_length=1, max_length=255)


class FolderOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: uuid.UUID
    tenant_id: uuid.UUID = Field(..., serialization_alias="tenantId")
    name: str
    is_deleted: bool = Field(..., serialization_alias="isDeleted")
    created_at: datetime = Field(..., serialization_alias="createdAt")
    updated_at: Optional[datetime] = Field(None, serialization_alias="updatedAt")


class AddFlowToFolderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    flow_id: uuid.UUID = Field(..., alias="flowId")


class FolderListResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    data: list[FolderOut]
    total: int
