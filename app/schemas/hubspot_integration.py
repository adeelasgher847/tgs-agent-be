from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, field_validator, model_validator
import re

_IDENTIFIER_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]{0,127}$')
_HUBSPOT_FIELD_RE = re.compile(r'^[a-z_][a-z0-9_]{0,127}$')


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

    @field_validator('hubspot_field')
    @classmethod
    def validate_hubspot_field(cls, v: str) -> str:
        if not _HUBSPOT_FIELD_RE.match(v):
            raise ValueError('hubspot_field must be a valid HubSpot property name (lowercase, underscores, max 128 chars)')
        return v

    @field_validator('prompt_variable')
    @classmethod
    def validate_prompt_variable(cls, v: str) -> str:
        if not _IDENTIFIER_RE.match(v):
            raise ValueError('prompt_variable must be a valid identifier (letters, digits, underscores, max 128 chars)')
        return v


class HubSpotFieldMappingRequest(BaseModel):
    mappings: List[HubSpotFieldMapping]

    @model_validator(mode='after')
    def validate_mapping_count(self) -> HubSpotFieldMappingRequest:
        if len(self.mappings) > 50:
            raise ValueError('A maximum of 50 field mappings are allowed')
        return self


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
