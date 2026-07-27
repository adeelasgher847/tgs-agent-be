from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class GhlAuthorizeResponse(BaseModel):
    authorization_url: str


class GhlContactOut(BaseModel):
    id: str
    name: str | None = None
    email: str | None = None
    tags: List[str] = []
    pipeline_stage: str | None = None
    last_activity_date: str | None = None


class GhlNoteCreateRequest(BaseModel):
    contact_id: str
    content: str


class GhlNoteCreateResponse(BaseModel):
    id: str | None = None
    contact_id: str


class GhlDisconnectResponse(BaseModel):
    disconnected: bool
    provider: str = "gohighlevel"


class GhlSettingsUpdateRequest(BaseModel):
    write_back_enabled: bool


class GhlIntegrationStatusOut(BaseModel):
    connected: bool
    connected_at: datetime | None = None
    last_sync_at: str | None = None
    write_back_enabled: bool = True


class GhlSyncStatusOut(BaseModel):
    last_lookup_at: str | None = None
    last_write_back_at: str | None = None
    last_write_back_status: str | None = None
    last_ghl_error: str | None = None
    error_count_24h: int = 0
