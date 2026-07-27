from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class CalendlyIntegrationStatusOut(BaseModel):
    connected: bool
    user_uri: Optional[str] = None
    event_type_uri: Optional[str] = None


class CalendlyDisconnectResponse(BaseModel):
    disconnected: bool
    provider: str = "calendly"
