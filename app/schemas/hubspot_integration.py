from __future__ import annotations

from datetime import datetime
from typing import List, Optional

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


class HubSpotFieldMapping(BaseModel):
    hubspot_field: str
    prompt_variable: str


class HubSpotFieldMappingRequest(BaseModel):
    mappings: List[HubSpotFieldMapping]


class HubSpotFieldMappingResponse(BaseModel):
    field_mappings: List[HubSpotFieldMapping]


class HubSpotSettingsUpdateRequest(BaseModel):
    contact_lookup_enabled: bool
    write_back_enabled: bool


class HubSpotIntegrationStatusOut(BaseModel):
    connected: bool
    connected_at: Optional[datetime] = None
    contact_lookup_enabled: bool = True
    write_back_enabled: bool = True
    field_mappings: List[HubSpotFieldMapping] = []
