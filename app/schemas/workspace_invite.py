from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr


class InviteCreate(BaseModel):
    email: EmailStr


class InviteOut(BaseModel):
    id: uuid.UUID
    email: str
    status: str
    expires_at: datetime
    created_at: datetime
    invited_by: uuid.UUID

    model_config = {"from_attributes": True}
