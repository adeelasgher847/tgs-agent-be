from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class MakeTriggerRequest(BaseModel):
    agent_id: str
    to_number: str
    variables: Optional[Dict[str, Any]] = None


class MakeTriggerResponse(BaseModel):
    call_id: str
    status: str


class N8nTriggerResponse(BaseModel):
    success: bool
    data: Dict[str, Any]


class IntegrationItem(BaseModel):
    name: str
    connected: bool
    webhook_url: str
    last_triggered_at: Optional[datetime] = None


class IntegrationListResponse(BaseModel):
    integrations: List[IntegrationItem]


class MakeSecretResponse(BaseModel):
    secret: str
    webhook_url: str
