from __future__ import annotations

import re
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


_E164ISH = re.compile(r"^\+[1-9]\d{6,14}$")


class TransferTypeEnum(str, Enum):
    cold = "cold"
    warm = "warm"


class TransferRouteBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    friendly_name: str = Field(..., min_length=1, max_length=255)
    phone_number: str = Field(..., min_length=8, max_length=20, description="E.164, e.g. +15551234567")
    transfer_type: TransferTypeEnum = Field(default=TransferTypeEnum.cold)

    @field_validator("phone_number")
    @classmethod
    def normalize_phone(cls, v: str) -> str:
        s = (v or "").strip()
        if not _E164ISH.match(s):
            raise ValueError("phone_number must be in E.164 format (e.g. +15551234567)")
        return s


class TransferRouteCreate(TransferRouteBase):
    pass


class TransferRouteUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    friendly_name: Optional[str] = Field(None, min_length=1, max_length=255)
    phone_number: Optional[str] = Field(None, min_length=8, max_length=20)
    transfer_type: Optional[TransferTypeEnum] = None

    @field_validator("phone_number")
    @classmethod
    def normalize_phone(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = v.strip()
        if not _E164ISH.match(s):
            raise ValueError("phone_number must be in E.164 format (e.g. +15551234567)")
        return s


class TransferRouteOut(TransferRouteBase):
    id: uuid.UUID
    tenant_id: uuid.UUID
    is_deleted: bool = False
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class TransferRouteListResponse(BaseModel):
    data: list[TransferRouteOut]
    total: int
