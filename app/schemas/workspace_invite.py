from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr


from typing import Optional


class InviteCreate(BaseModel):
    email: EmailStr
    role_id: Optional[uuid.UUID] = None


class InviteOut(BaseModel):
    id: uuid.UUID
    email: str
    status: str
    expires_at: datetime
    created_at: datetime
    invited_by: uuid.UUID
    role_id: Optional[uuid.UUID] = None

    model_config = {"from_attributes": True}

