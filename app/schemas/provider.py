"""
Provider schemas for API serialization
"""

from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class ProviderBase(BaseModel):
    """Base provider schema with common fields"""
    name: str = Field(..., max_length=100, description="Name of the AI provider (e.g., OpenAI, Google, Anthropic)")
    api_key: Optional[str] = Field(None, max_length=500, description="API key for the provider (encrypted)")
    is_active: bool = Field(True, description="Whether the provider is active")


class ProviderCreate(ProviderBase):
    """Schema for creating a new provider"""
    pass


class ProviderUpdate(BaseModel):
    """Schema for updating a provider"""
    name: Optional[str] = Field(None, max_length=100)
    api_key: Optional[str] = Field(None, max_length=500)
    is_active: Optional[bool] = None


class ProviderResponse(ProviderBase):
    """Schema for provider responses"""
    id: uuid.UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class ProviderList(BaseModel):
    """Schema for listing providers"""
    providers: list[ProviderResponse]
    total: int
    page: int
    size: int
