from __future__ import annotations


from pydantic import BaseModel


class CalendlyIntegrationStatusOut(BaseModel):
    connected: bool
    user_uri: str | None = None
    event_type_uri: str | None = None


class CalendlyDisconnectResponse(BaseModel):
    disconnected: bool
    provider: str = "calendly"
