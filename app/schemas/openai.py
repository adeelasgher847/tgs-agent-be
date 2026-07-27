"""
OpenAI API schemas for text generation and chat completions
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid


class OpenAITestRequest(BaseModel):
    """Schema for testing OpenAI text generation"""
    model_id: uuid.UUID = Field(..., description="ID of the model to use for generation")
    prompt: str = Field(..., min_length=1, max_length=10000, description="Input prompt for text generation")
    system_prompt: Optional[str] = Field(None, max_length=2000, description="System prompt to set context")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Temperature setting (0.0 to 2.0)")
    max_tokens: Optional[int] = Field(1000, gt=0, le=16384, description="Maximum tokens for response")


class OpenAITestResponse(BaseModel):
    """Schema for OpenAI text generation response"""
    content: str = Field(..., description="Generated text content")
    model_name: str = Field(..., description="Name of the model used")
    response_time: float = Field(..., description="Response time in seconds")
    usage: Dict[str, int] = Field(..., description="Token usage information")
    finish_reason: str = Field(..., description="Reason for completion")


class OpenAIChatRequest(BaseModel):
    """Schema for OpenAI chat completion"""
    model_id: uuid.UUID = Field(..., description="ID of the model to use for generation")
    messages: List[Dict[str, str]] = Field(..., description="List of conversation messages")
    system_prompt: Optional[str] = Field(None, max_length=2000, description="System prompt to set context")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Temperature setting (0.0 to 2.0)")
    max_tokens: Optional[int] = Field(1000, gt=0, le=16384, description="Maximum tokens for response")


class OpenAIChatResponse(BaseModel):
    """Schema for OpenAI chat completion response"""
    content: str = Field(..., description="Generated text content")
    model_name: str = Field(..., description="Name of the model used")
    response_time: float = Field(..., description="Response time in seconds")
    usage: Dict[str, int] = Field(..., description="Token usage information")
    finish_reason: str = Field(..., description="Reason for completion")

