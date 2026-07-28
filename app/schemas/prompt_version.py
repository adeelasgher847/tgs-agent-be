from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PromptVersionOut(BaseModel):
    """API projection of a prompt version — gemini_prompt is never included."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: uuid.UUID
    flow_id: uuid.UUID = Field(..., serialization_alias="flowId")
    prompt_text: str = Field(..., serialization_alias="promptText")
    notes: str | None = None
    created_at: datetime = Field(..., serialization_alias="createdAt")
