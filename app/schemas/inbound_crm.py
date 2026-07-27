"""
Schemas for tenant inbound call log → CRM (Trello) configuration.
"""

from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class TenantInboundCRMConfigPublic(BaseModel):
    """Safe for any tenant member — no secrets."""

    id: UUID
    tenant_id: UUID
    provider: str
    connection_type: str
    container_id: Optional[str] = None
    container_url: Optional[str] = None
    is_enabled: bool
    has_credentials: bool = False

    model_config = ConfigDict(from_attributes=True)


class TenantInboundCRMConfigUpsert(BaseModel):
    """Owner-only: BYO Trello or enable platform board."""

    provider: str = Field(default="trello", max_length=20)
    connection_type: str = Field(default="byo_credentials", max_length=30)
    api_key: Optional[str] = None
    api_token: Optional[str] = None
    container_id: Optional[str] = None
    board_url: Optional[str] = None
    is_enabled: bool = True
    extra_config: Optional[Dict[str, Any]] = None


class TenantInboundCRMProvisionResponse(BaseModel):
    board_id: str
    board_url: str
    list_id: str


class InboundBoardUrlOut(BaseModel):
    """Public board link — no secrets."""

    board_url: str
    board_id: str
