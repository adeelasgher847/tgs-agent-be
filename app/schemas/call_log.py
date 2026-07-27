from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

class CallLogBase(BaseModel):
    call_session_id: uuid.UUID
    tenant_id: uuid.UUID
    call_id: str = Field(description="Shortened call ID for display")
    external_call_id: str | None = None
    call_type: str = Field(default="inbound", description="inbound, outbound, web")
    success_evaluation: str | None = Field(None, description="success, fail, null")
    ended_reason: str | None = None
    transferred: bool = Field(default=False, description="Whether call was transferred")
    assistant_phone_number: str | None = None
    customer_phone_number: str | None = None
    cost: float | None = Field(default=0.0, description="Cost in USD")
    cost_currency: str | None = Field(default="USD", description="Currency code")
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration: int | None = Field(description="Duration in seconds")
    call_metadata: Dict[str, Any] | None = None
    notes: str | None = None

class CallLogCreate(BaseModel):
    call_session_id: uuid.UUID
    tenant_id: uuid.UUID
    call_id: str
    external_call_id: str | None = None
    call_type: str = Field(default="inbound", description="inbound, outbound, web")
    success_evaluation: str | None = None
    ended_reason: str | None = None
    transferred: bool = Field(default=False)
    assistant_phone_number: str | None = None
    customer_phone_number: str | None = None
    cost: float | None = Field(default=0.0)
    cost_currency: str | None = Field(default="USD")
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration: int | None = None
    call_metadata: Dict[str, Any] | None = None
    notes: str | None = None

class CallLogUpdate(BaseModel):
    success_evaluation: str | None = None
    ended_reason: str | None = None
    transferred: bool | None = None
    cost: float | None = None
    end_time: datetime | None = None
    duration: int | None = None
    call_metadata: Dict[str, Any] | None = None
    notes: str | None = None

class CallLogResponse(CallLogBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime | None = None

# Dashboard-specific schemas
class CallLogDashboardResponse(BaseModel):
    """Call log response model for dashboard display (like Vapi)"""
    id: uuid.UUID
    call_id: str = Field(description="Shortened call ID for display")
    assistant_name: str = Field(description="Name of the assistant")
    assistant_phone_number: str | None = None
    customer_phone_number: str | None = None
    call_type: str = Field(description="inbound, outbound, web")
    ended_reason: str | None = None
    success_evaluation: str | None = None
    start_time: datetime | None = None
    duration: int | None = Field(description="Duration in seconds")
    cost: float | None = Field(description="Cost in USD")
    transferred: bool = False
    created_at: datetime

class CallLogFilters(BaseModel):
    """Filters for call logs query"""
    call_type: str | None = None  # inbound, outbound, web
    success_evaluation: str | None = None  # success, fail
    agent_id: uuid.UUID | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    transferred: bool | None = None
    ended_reason: str | None = None
    assistant_phone_number: str | None = None
    customer_phone_number: str | None = None

class CallLogStats(BaseModel):
    """Statistics for call logs dashboard"""
    total_calls: int
    successful_calls: int
    failed_calls: int
    transferred_calls: int
    total_cost: float
    average_duration: float | None = None
    calls_by_type: Dict[str, int] = Field(default_factory=dict)
    calls_by_agent: Dict[str, int] = Field(default_factory=dict)
    calls_by_ended_reason: Dict[str, int] = Field(default_factory=dict)

class CallLogList(BaseModel):
    """Paginated call logs response"""
    logs: List[CallLogDashboardResponse]
    total: int
    stats: CallLogStats
    page: int
    per_page: int

class CallLogExport(BaseModel):
    """Call log export format"""
    call_id: str
    assistant_name: str
    assistant_phone_number: str | None
    customer_phone_number: str | None
    call_type: str
    ended_reason: str | None
    success_evaluation: str | None
    start_time: datetime | None
    duration: int | None
    cost: float | None
    transferred: bool
    created_at: datetime
