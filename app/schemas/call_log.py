from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

class CallLogBase(BaseModel):
    call_session_id: uuid.UUID
    tenant_id: uuid.UUID
    call_id: str = Field(description="Shortened call ID for display")
    external_call_id: Optional[str] = None
    call_type: str = Field(default="inbound", description="inbound, outbound, web")
    success_evaluation: Optional[str] = Field(None, description="success, fail, null")
    ended_reason: Optional[str] = None
    transferred: bool = Field(default=False, description="Whether call was transferred")
    assistant_phone_number: Optional[str] = None
    customer_phone_number: Optional[str] = None
    cost: Optional[float] = Field(default=0.0, description="Cost in USD")
    cost_currency: Optional[str] = Field(default="USD", description="Currency code")
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[int] = Field(description="Duration in seconds")
    call_metadata: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

class CallLogCreate(BaseModel):
    call_session_id: uuid.UUID
    tenant_id: uuid.UUID
    call_id: str
    external_call_id: Optional[str] = None
    call_type: str = Field(default="inbound", description="inbound, outbound, web")
    success_evaluation: Optional[str] = None
    ended_reason: Optional[str] = None
    transferred: bool = Field(default=False)
    assistant_phone_number: Optional[str] = None
    customer_phone_number: Optional[str] = None
    cost: Optional[float] = Field(default=0.0)
    cost_currency: Optional[str] = Field(default="USD")
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[int] = None
    call_metadata: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

class CallLogUpdate(BaseModel):
    success_evaluation: Optional[str] = None
    ended_reason: Optional[str] = None
    transferred: Optional[bool] = None
    cost: Optional[float] = None
    end_time: Optional[datetime] = None
    duration: Optional[int] = None
    call_metadata: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

class CallLogResponse(CallLogBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: Optional[datetime] = None

# Dashboard-specific schemas
class CallLogDashboardResponse(BaseModel):
    """Call log response model for dashboard display (like Vapi)"""
    id: uuid.UUID
    call_id: str = Field(description="Shortened call ID for display")
    assistant_name: str = Field(description="Name of the assistant")
    assistant_phone_number: Optional[str] = None
    customer_phone_number: Optional[str] = None
    call_type: str = Field(description="inbound, outbound, web")
    ended_reason: Optional[str] = None
    success_evaluation: Optional[str] = None
    start_time: Optional[datetime] = None
    duration: Optional[int] = Field(description="Duration in seconds")
    cost: Optional[float] = Field(description="Cost in USD")
    transferred: bool = False
    created_at: datetime

class CallLogFilters(BaseModel):
    """Filters for call logs query"""
    call_type: Optional[str] = None  # inbound, outbound, web
    success_evaluation: Optional[str] = None  # success, fail
    agent_id: Optional[uuid.UUID] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    transferred: Optional[bool] = None
    ended_reason: Optional[str] = None
    assistant_phone_number: Optional[str] = None
    customer_phone_number: Optional[str] = None

class CallLogStats(BaseModel):
    """Statistics for call logs dashboard"""
    total_calls: int
    successful_calls: int
    failed_calls: int
    transferred_calls: int
    total_cost: float
    average_duration: Optional[float] = None
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
    assistant_phone_number: Optional[str]
    customer_phone_number: Optional[str]
    call_type: str
    ended_reason: Optional[str]
    success_evaluation: Optional[str]
    start_time: Optional[datetime]
    duration: Optional[int]
    cost: Optional[float]
    transferred: bool
    created_at: datetime
