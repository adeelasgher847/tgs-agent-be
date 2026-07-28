from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.call_flow import CallFlowListItem


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
    updated_at: datetime | None = Field(None, serialization_alias="updatedAt")


class AddFlowToFolderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    flow_id: uuid.UUID = Field(..., alias="flowId")


class FolderListResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    data: list[FolderOut]
    total: int


class FolderFlowsResponse(BaseModel):
    """Response body for ``GET /api/v1/folders/{folder_id}/flows``."""

    model_config = ConfigDict(populate_by_name=True)

    folder_id: uuid.UUID = Field(..., serialization_alias="folderId")
    data: list[CallFlowListItem]
    total: int
