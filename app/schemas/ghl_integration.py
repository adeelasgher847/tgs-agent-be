from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class GhlAuthorizeResponse(BaseModel):
    authorization_url: str


class GhlContactOut(BaseModel):
    id: str
    name: Optional[str] = None
    email: Optional[str] = None
    tags: List[str] = []
    pipeline_stage: Optional[str] = None
    last_activity_date: Optional[str] = None


class GhlNoteCreateRequest(BaseModel):
    contact_id: str
    content: str


class GhlNoteCreateResponse(BaseModel):
    id: Optional[str] = None
    contact_id: str


class GhlDisconnectResponse(BaseModel):
    disconnected: bool
    provider: str = "gohighlevel"


class GhlSettingsUpdateRequest(BaseModel):
    write_back_enabled: bool


class GhlIntegrationStatusOut(BaseModel):
    connected: bool
    connected_at: Optional[datetime] = None
    last_sync_at: Optional[str] = None
    write_back_enabled: bool = True


class GhlSyncStatusOut(BaseModel):
    last_lookup_at: Optional[str] = None
    last_write_back_at: Optional[str] = None
    last_write_back_status: Optional[str] = None
    last_ghl_error: Optional[str] = None
    error_count_24h: int = 0
