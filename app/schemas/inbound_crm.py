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
    container_id: str | None = None
    container_url: str | None = None
    is_enabled: bool
    has_credentials: bool = False

    model_config = ConfigDict(from_attributes=True)


class TenantInboundCRMConfigUpsert(BaseModel):
    """Owner-only: BYO Trello or enable platform board."""

    provider: str = Field(default="trello", max_length=20)
    connection_type: str = Field(default="byo_credentials", max_length=30)
    api_key: str | None = None
    api_token: str | None = None
    container_id: str | None = None
    board_url: str | None = None
    is_enabled: bool = True
    extra_config: Dict[str, Any] | None = None


class TenantInboundCRMProvisionResponse(BaseModel):
    board_id: str
    board_url: str
    list_id: str


class InboundBoardUrlOut(BaseModel):
    """Public board link — no secrets."""

    board_url: str
    board_id: str
