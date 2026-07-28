from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel


class MakeTriggerRequest(BaseModel):
    agent_id: str
    to_number: str
    variables: Dict[str, Any] | None = None


class MakeTriggerResponse(BaseModel):
    call_id: str
    status: str


class N8nTriggerResponse(BaseModel):
    success: bool
    data: Dict[str, Any]


class IntegrationItem(BaseModel):
    name: str
    connected: bool
    webhook_url: str | None = None
    last_triggered_at: datetime | None = None
    connected_at: datetime | None = None
    last_sync_at: str | None = None


class IntegrationListResponse(BaseModel):
    integrations: List[IntegrationItem]


class MakeSecretResponse(BaseModel):
    secret: str
    webhook_url: str


class N8nSecretResponse(BaseModel):
    secret: str
    webhook_url: str
