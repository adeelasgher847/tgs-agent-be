from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
import uuid

class CallSessionBase(BaseModel):
    user_id: uuid.UUID
    agent_id: uuid.UUID
    tenant_id: uuid.UUID
    status: str
    call_type: str = Field(default="inbound", description="inbound, outbound, web")
    success_evaluation: Optional[str] = Field(None, description="success, fail, null")
    ended_reason: Optional[str] = None
    cost: Optional[float] = Field(default=0.0, description="Cost in USD")
    cost_currency: Optional[str] = Field(default="USD", description="Currency code")
    transferred: bool = Field(default=False, description="Whether call was transferred")
    twilio_call_sid: Optional[str] = None
    from_number: Optional[str] = None
    to_number: Optional[str] = None
    assistant_phone_number: Optional[str] = None
    customer_phone_number: Optional[str] = None
    call_metadata: Optional[Dict[str, Any]] = None

class CallSessionCreate(BaseModel):
    user_id: uuid.UUID
    agent_id: uuid.UUID
    tenant_id: uuid.UUID
    call_type: str = Field(default="inbound", description="inbound, outbound, web")
    twilio_call_sid: Optional[str] = None
    from_number: Optional[str] = None
    to_number: Optional[str] = None
    assistant_phone_number: Optional[str] = None
    customer_phone_number: Optional[str] = None
    call_metadata: Optional[Dict[str, Any]] = None

class CallSessionUpdate(BaseModel):
    status: Optional[str] = None
    end_time: Optional[datetime] = None
    duration: Optional[int] = None
    success_evaluation: Optional[str] = None
    ended_reason: Optional[str] = None
    cost: Optional[float] = None
    transferred: Optional[bool] = None
    call_transcript: Optional[List[Dict[str, Any]]] = None
    response_times: Optional[List[Dict[str, Any]]] = None
    call_metadata: Optional[Dict[str, Any]] = None

class CallSessionResponse(CallSessionBase):
    id: uuid.UUID
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[int] = None
    call_transcript: Optional[List[Dict[str, Any]]] = None
    response_times: Optional[List[Dict[str, Any]]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

class CallSessionStats(BaseModel):
    session_id: str
    status: str
    duration: Optional[int] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    total_messages: int
    user_messages: int
    assistant_messages: int
    average_response_time: Optional[float] = None
    total_response_time_entries: int

class TranscriptEntry(BaseModel):
    timestamp: str
    role: str
    content: str

class ResponseTimeEntry(BaseModel):
    timestamp: str
    response_time: float

class CallSessionList(BaseModel):
    sessions: List[CallSessionResponse]
    total: int

# Call Logs specific schemas for dashboard-like interface
class CallLogResponse(BaseModel):
    """Call log response model matching Vapi dashboard structure"""
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

class CallLogList(BaseModel):
    """Paginated call logs response"""
    logs: List[CallLogResponse]
    total: int
    stats: CallLogStats
    page: int
    per_page: int


class CallLogAnalysisEmailRequest(BaseModel):
    """Request payload for sending a call-related email based on a call session.

    - Backend always generates an analysis from the call transcript.
    - If transform_prompt is provided, AI uses it to create a custom email.
    - If transform_prompt is missing, the analysis is forwarded as the email body.
    """

    call_session_id: uuid.UUID = Field(..., description="Call session whose data will be used.")
    target_email: EmailStr = Field(..., description="Recipient email address.")

    transform_prompt: Optional[str] = Field(
        None,
        description=(
            "Optional custom instruction for AI. If provided, the model will use the call analysis and this prompt "
            "to generate the email body. If omitted, the analysis text itself will be sent as the email."
        ),
    )
