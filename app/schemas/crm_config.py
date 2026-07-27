"""
Schemas for CRM Configuration
"""

from pydantic import BaseModel
from uuid import UUID


class CRMConfigCreate(BaseModel):
    """Schema for creating CRM configuration"""
    crm_type: str  # "monday" | "clickup" | "jira" | "trello"
    api_key: str | None = None  # Plain API key (will be encrypted). Optional for ClickUp OAuth (will be set after OAuth)
    container_id: str | None = None  # board_id/list_id/project_id (optional, can be created)
    container_url: str | None = None
    additional_config: dict | None = None  # CRM-specific config (workspace_id, email, server_url, etc.)


class CRMConfigUpdate(BaseModel):
    """Schema for updating CRM configuration"""
    api_key: str | None = None
    container_id: str | None = None
    container_url: str | None = None
    additional_config: dict | None = None


class CRMConfigOut(BaseModel):
    """Schema for CRM configuration response"""
    id: UUID
    crm_type: str
    container_id: str | None
    container_url: str | None
    additional_config: dict | None
    created_at: str
    updated_at: str | None
    
    class Config:
        from_attributes = True


class CRMConfigResponse(BaseModel):
    """Response schema for GET /scheduled-calls/crm-config"""
    crm_config_id: str
    crm_type: str
    crm_container_id: str | None
    crm_container_url: str | None


class CRMConfigListItem(BaseModel):
    """Schema for CRM config list item"""
    id: str
    crm_type: str
    crm_type_display: str  # Display name like "Monday.com", "ClickUp", etc.
    container_id: str | None
    container_url: str | None
    created_at: str


class CRMConfigListResponse(BaseModel):
    """Response schema for GET /scheduled-calls/crm-config - list of all configured CRMs"""
    configured_crms: list[CRMConfigListItem]

