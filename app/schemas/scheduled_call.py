from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import uuid

class ScheduledCallBase(BaseModel):
    phone_number: str = Field(..., description="Phone number to call")
    agent_id: uuid.UUID = Field(..., description="Agent ID for the call")
    scheduled_time_utc: datetime = Field(..., description="Scheduled time in UTC")
    status: str = Field(default="pending", description="Status: pending, scheduled, failed, completed")

class ScheduledCallCreate(BaseModel):
    phone_number: str
    agent_id: uuid.UUID
    scheduled_time_utc: datetime
    status: str = "pending"

class ScheduledCallResponse(ScheduledCallBase):
    id: uuid.UUID
    tenant_id: uuid.UUID
    user_id: uuid.UUID
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class ScheduledCallList(BaseModel):
    calls: List[ScheduledCallResponse]
    total: int
    skip: int = Field(default=0, description="Number of records skipped")
    limit: int = Field(default=50, description="Maximum records per page")

class ScheduledCallUpdate(BaseModel):
    status: str = Field(..., description="Status to update: pending, scheduled, failed, completed")

class CSVUploadResponse(BaseModel):
    total_rows: int
    successful_rows: int
    failed_rows: int
    errors: List[str] = Field(default_factory=list)

