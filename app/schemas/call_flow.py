from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.prompt_version import PromptVersionOut


class DirectionEnum(str, Enum):
    inbound = "inbound"
    outbound = "outbound"
    bidirectional = "bidirectional"


class WelcomeMessageTypeEnum(str, Enum):
    user_initiated = "user_initiated"
    ai_dynamic = "ai_dynamic"
    ai_custom = "ai_custom"


class FlowDataSchema(BaseModel):
    """Structural validation for flowData JSONB — future visual-editor format."""

    model_config = ConfigDict(extra="allow")

    nodes: List[Any] = Field(default_factory=list)
    edges: List[Any] = Field(default_factory=list)


class AgentRef(BaseModel):
    """Embedded agent snapshot in flow responses."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str


class CallFlowCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(..., min_length=1, max_length=255)
    direction: DirectionEnum
    agent_id: uuid.UUID = Field(..., alias="agentId")
    welcome_message_type: Optional[WelcomeMessageTypeEnum] = Field(None, alias="welcomeMessageType")
    custom_welcome_message: Optional[str] = Field(None, alias="customWelcomeMessage")
    prompt: Optional[str] = None
    notes: Optional[str] = None  # notes for the initial prompt version
    flow_data: Optional[FlowDataSchema] = Field(None, alias="flowData")
    settings: Optional[Dict[str, Any]] = None


class CallFlowUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    direction: Optional[DirectionEnum] = None
    agent_id: Optional[uuid.UUID] = Field(None, alias="agentId")
    welcome_message_type: Optional[WelcomeMessageTypeEnum] = Field(None, alias="welcomeMessageType")
    custom_welcome_message: Optional[str] = Field(None, alias="customWelcomeMessage")
    prompt: Optional[str] = None
    notes: Optional[str] = None  # notes for the new prompt version
    current_prompt_id: Optional[uuid.UUID] = Field(None, alias="currentPromptId")
    flow_data: Optional[FlowDataSchema] = Field(None, alias="flowData")
    settings: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def prompt_and_rollback_exclusive(self) -> "CallFlowUpdate":
        if self.prompt and self.current_prompt_id:
            raise ValueError(
                "'prompt' and 'currentPromptId' are mutually exclusive — "
                "use 'prompt' to create a new version or 'currentPromptId' to roll back, not both."
            )
        return self


class CallFlowSettingsUpdate(BaseModel):
    """Request body for ``PUT /api/v1/call-flows/{id}/settings``."""

    model_config = ConfigDict(extra="forbid")

    public_access: bool = Field(..., alias="public_access")


class CallerMemorySettingsUpdate(BaseModel):
    """Request body for ``PUT /api/v2/flows/{flow_id}/caller-memory-settings``."""

    model_config = ConfigDict(extra="forbid")

    caller_memory_enabled: bool
    caller_memory_window: int = Field(..., ge=1, le=10)


class CallerMemorySettingsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    caller_memory_enabled: bool
    caller_memory_window: int


class CallFlowOut(BaseModel):
    """Full flow response including all prompt versions."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: uuid.UUID
    name: str
    direction: str
    agent_id: uuid.UUID = Field(..., serialization_alias="agentId")
    # Full AgentOut shape on detail endpoints; slim AgentRef on list
    agent: Optional[Dict[str, Any]] = None
    welcome_message_type: Optional[str] = Field(None, serialization_alias="welcomeMessageType")
    custom_welcome_message: Optional[str] = Field(None, serialization_alias="customWelcomeMessage")
    current_prompt_id: Optional[uuid.UUID] = Field(None, serialization_alias="currentPromptId")
    prompt_versions: List[PromptVersionOut] = Field(default_factory=list, serialization_alias="promptVersions")
    flow_data: Optional[Dict[str, Any]] = Field(None, serialization_alias="flowData")
    settings: Optional[Dict[str, Any]] = None
    public_access: bool = Field(False, serialization_alias="publicAccess")
    created_at: datetime = Field(..., serialization_alias="createdAt")
    updated_at: Optional[datetime] = Field(None, serialization_alias="updatedAt")


class CallFlowListItem(BaseModel):
    """Slim flow item used in the paginated list — no prompt_versions payload."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: uuid.UUID
    name: str
    direction: str
    agent_id: uuid.UUID = Field(..., serialization_alias="agentId")
    agent: Optional[AgentRef] = None
    welcome_message_type: Optional[str] = Field(None, serialization_alias="welcomeMessageType")
    custom_welcome_message: Optional[str] = Field(None, serialization_alias="customWelcomeMessage")
    current_prompt_id: Optional[uuid.UUID] = Field(None, serialization_alias="currentPromptId")
    flow_data: Optional[Dict[str, Any]] = Field(None, serialization_alias="flowData")
    settings: Optional[Dict[str, Any]] = None
    public_access: bool = Field(False, serialization_alias="publicAccess")
    created_at: datetime = Field(..., serialization_alias="createdAt")
    updated_at: Optional[datetime] = Field(None, serialization_alias="updatedAt")


class CallFlowListResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    data: List[CallFlowListItem]
    total: int
    page: int
    page_size: int = Field(..., serialization_alias="pageSize")
