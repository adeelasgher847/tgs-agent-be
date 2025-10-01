"""
Model schemas for API serialization
"""

from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class ModelBase(BaseModel):
    """Base model schema with common fields"""
    model_name: str = Field(..., max_length=100, description="Name of the AI model")
    api_key: Optional[str] = Field(None, max_length=500, description="Model-specific API key")
    description: Optional[str] = Field(None, description="Model description including free tokens, efficiency, pricing details")
    system_prompt: Optional[str] = Field(None, max_length=1000, description="Default system prompt for the model")
    temperature: Optional[int] = Field(None, ge=0, le=100, description="Temperature setting (0-100)")
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum tokens for responses")
    archive: bool = Field(True, description="Whether the model is archived")


class ModelCreate(ModelBase):
    """Schema for creating a new model"""
    provider_id: uuid.UUID = Field(..., description="ID of the provider")


class ModelUpdate(BaseModel):
    """Schema for updating a model"""
    model_name: Optional[str] = Field(None, max_length=100)
    api_key: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = None
    system_prompt: Optional[str] = Field(None, max_length=1000)
    temperature: Optional[int] = Field(None, ge=0, le=100)
    max_tokens: Optional[int] = Field(None, gt=0)
    archive: Optional[bool] = None


class ModelResponse(BaseModel):
    """Schema for model responses (API key excluded for security)"""
    id: uuid.UUID
    provider_id: uuid.UUID
    model_name: str = Field(..., max_length=100, description="Name of the AI model")
    description: Optional[str] = Field(None, description="Model description including free tokens, efficiency, pricing details")
    system_prompt: Optional[str] = Field(None, max_length=1000, description="Default system prompt for the model")
    temperature: Optional[int] = Field(None, ge=0, le=100, description="Temperature setting (0-100)")
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum tokens for responses")
    archive: bool = Field(True, description="Whether the model is archived")
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class ModelList(BaseModel):
    """Schema for listing models"""
    models: list[ModelResponse]
    total: int
    page: int
    size: int
