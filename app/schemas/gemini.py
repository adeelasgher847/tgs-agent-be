"""
Gemini API schemas for text generation
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid


class GeminiTestRequest(BaseModel):
    """Schema for testing Gemini text generation"""
    model_id: uuid.UUID = Field(..., description="ID of the model to use for generation")
    prompt: str = Field(..., min_length=1, max_length=10000, description="Input prompt for text generation")
    system_prompt: Optional[str] = Field(None, max_length=2000, description="System prompt to set context")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Temperature setting (0.0 to 1.0)")
    max_tokens: Optional[int] = Field(1000, gt=0, le=8192, description="Maximum tokens for response")


class GeminiTestResponse(BaseModel):
    """Schema for Gemini text generation response"""
    content: str = Field(..., description="Generated text content")
    model_name: str = Field(..., description="Name of the model used")
    response_time: float = Field(..., description="Response time in seconds")
    usage: Dict[str, int] = Field(..., description="Token usage information")
    finish_reason: str = Field(..., description="Reason for completion")


class GeminiChatRequest(BaseModel):
    """Schema for Gemini chat completion"""
    model_id: uuid.UUID = Field(..., description="ID of the model to use for generation")
    messages: List[Dict[str, str]] = Field(..., description="List of conversation messages")
    system_prompt: Optional[str] = Field(None, max_length=2000, description="System prompt to set context")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Temperature setting (0.0 to 1.0)")
    max_tokens: Optional[int] = Field(1000, gt=0, le=8192, description="Maximum tokens for response")


class GeminiChatResponse(BaseModel):
    """Schema for Gemini chat completion response"""
    content: str = Field(..., description="Generated text content")
    model_name: str = Field(..., description="Name of the model used")
    response_time: float = Field(..., description="Response time in seconds")
    usage: Dict[str, int] = Field(..., description="Token usage information")
    finish_reason: str = Field(..., description="Reason for completion")
