"""
Schemas for CRM Configuration
"""

from pydantic import BaseModel
from typing import Optional
from uuid import UUID


class TenantCRMConfigCreate(BaseModel):
    """Schema for creating CRM configuration"""
    crm_type: str  # "monday" | "clickup" | "jira" | "trello"
    api_key: str  # Plain API key (will be encrypted)
    container_id: Optional[str] = None  # board_id/list_id/project_id (optional, can be created)
    container_url: Optional[str] = None
    additional_config: Optional[dict] = None  # CRM-specific config (workspace_id, email, server_url, etc.)


class TenantCRMConfigUpdate(BaseModel):
    """Schema for updating CRM configuration"""
    api_key: Optional[str] = None
    container_id: Optional[str] = None
    container_url: Optional[str] = None
    additional_config: Optional[dict] = None


class TenantCRMConfigOut(BaseModel):
    """Schema for CRM configuration response"""
    id: UUID
    tenant_id: UUID
    crm_type: str
    container_id: Optional[str]
    container_url: Optional[str]
    additional_config: Optional[dict]
    created_at: str
    updated_at: Optional[str]
    
    class Config:
        from_attributes = True


class CRMConfigResponse(BaseModel):
    """Response schema for GET /scheduled-calls/crm-config"""
    crm_config_id: str
    crm_type: str
    crm_container_id: Optional[str]
    crm_container_url: Optional[str]


class CRMConfigListItem(BaseModel):
    """Schema for CRM config list item"""
    id: str
    crm_type: str
    crm_type_display: str  # Display name like "Monday.com", "ClickUp", etc.
    container_id: Optional[str]
    container_url: Optional[str]
    created_at: str


class CRMConfigListResponse(BaseModel):
    """Response schema for GET /scheduled-calls/crm-config - list of all configured CRMs"""
    configured_crms: list[CRMConfigListItem]

