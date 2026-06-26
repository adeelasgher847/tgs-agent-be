from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class HubSpotAuthorizeResponse(BaseModel):
    authorization_url: str


class HubSpotContactOut(BaseModel):
    id: str
    name: Optional[str] = None
    email: Optional[str] = None
    company: Optional[str] = None
    last_interaction_date: Optional[str] = None


class HubSpotDisconnectResponse(BaseModel):
    disconnected: bool
    provider: str = "hubspot"
