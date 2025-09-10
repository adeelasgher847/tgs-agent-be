from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

class CallSessionBase(BaseModel):
    user_id: uuid.UUID
    agent_id: uuid.UUID
    tenant_id: uuid.UUID
    status: str
    twilio_call_sid: Optional[str] = None
    from_number: Optional[str] = None
    to_number: Optional[str] = None

class CallSessionCreate(BaseModel):
    user_id: uuid.UUID
    agent_id: uuid.UUID
    tenant_id: uuid.UUID
    twilio_call_sid: Optional[str] = None
    from_number: Optional[str] = None
    to_number: Optional[str] = None

class CallSessionUpdate(BaseModel):
    status: Optional[str] = None
    end_time: Optional[datetime] = None
    duration: Optional[int] = None

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
