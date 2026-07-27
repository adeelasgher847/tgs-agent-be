from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class SalesforceAuthorizeResponse(BaseModel):
    authorization_url: str


class SalesforceContactOut(BaseModel):
    id: str
    name: str | None = None
    account: str | None = None
    email: str | None = None


class SalesforceDisconnectResponse(BaseModel):
    disconnected: bool
    provider: str = "salesforce"


class SalesforceSettingsUpdateRequest(BaseModel):
    write_back_enabled: bool


class SalesforceIntegrationStatusOut(BaseModel):
    connected: bool
    connected_at: datetime | None = None
    last_sync_at: str | None = None
    write_back_enabled: bool = True


class SalesforceSyncStatusOut(BaseModel):
    last_lookup_at: str | None = None
    last_write_back_at: str | None = None
    last_write_back_status: str | None = None
    error_count_24h: int = 0
