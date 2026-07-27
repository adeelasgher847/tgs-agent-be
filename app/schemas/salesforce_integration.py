from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class SalesforceAuthorizeResponse(BaseModel):
    authorization_url: str


class SalesforceContactOut(BaseModel):
    id: str
    name: Optional[str] = None
    account: Optional[str] = None
    email: Optional[str] = None


class SalesforceDisconnectResponse(BaseModel):
    disconnected: bool
    provider: str = "salesforce"


class SalesforceSettingsUpdateRequest(BaseModel):
    write_back_enabled: bool


class SalesforceIntegrationStatusOut(BaseModel):
    connected: bool
    connected_at: Optional[datetime] = None
    last_sync_at: Optional[str] = None
    write_back_enabled: bool = True


class SalesforceSyncStatusOut(BaseModel):
    last_lookup_at: Optional[str] = None
    last_write_back_at: Optional[str] = None
    last_write_back_status: Optional[str] = None
    error_count_24h: int = 0
