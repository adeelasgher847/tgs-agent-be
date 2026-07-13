"""Recording API schemas — Pydantic v2."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class RecordingResponse(BaseModel):
    """Response for GET /api/v1/recordings/{call_id}."""

    url: str = Field(..., description="GCS signed URL (expires in 1 hour)")
    duration: Optional[int] = Field(None, description="Call duration in seconds")
    size: Optional[int] = Field(None, description="Recording file size in bytes")
