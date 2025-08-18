from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid


# Call-related schemas
class CallInfo(BaseModel):
    sid: str
    from_number: str
    to: str
    status: str
    start_time: Optional[str] = None
    duration: Optional[str] = None


class CallsResponse(BaseModel):
    calls: List[CallInfo]


class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None


# Phone number management schemas
class AvailableNumberInfo(BaseModel):
    phone_number: str
    friendly_name: Optional[str] = None
    locality: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None
    capabilities: Dict[str, bool]
    beta: bool


class AvailableNumbersResponse(BaseModel):
    numbers: List[AvailableNumberInfo]
    total: int


class PhoneNumberInfo(BaseModel):
    sid: str
    phone_number: str
    friendly_name: Optional[str] = None
    voice_url: Optional[str] = None
    voice_method: Optional[str] = None
    status_callback: Optional[str] = None
    status_callback_method: Optional[str] = None
    capabilities: Dict[str, bool]
    date_created: str
    date_updated: str


class PhoneNumbersResponse(BaseModel):
    numbers: List[PhoneNumberInfo]
    total: int


class PurchaseNumberRequest(BaseModel):
    phone_number: str
    webhook_url: Optional[str] = None
    status_callback_url: Optional[str] = None
    status_callback_method: str = "POST"


class UpdateNumberRequest(BaseModel):
    friendly_name: Optional[str] = None
    webhook_url: Optional[str] = None
    status_callback_url: Optional[str] = None


class AccountInfo(BaseModel):
    sid: str
    friendly_name: str
    status: str
    type: str
    date_created: str
    date_updated: str


class CallResponse(BaseModel):
    success: bool
    call_sid: str
    to_number: str
    from_number: str
    status: str
    message: Optional[str] = None


class MakeCallRequest(BaseModel):
    to_number: str
    webhook_url: Optional[str] = None
    status_callback_url: Optional[str] = None


# Voice call initiation schemas
class CallInitiateRequest(BaseModel):
    agentId: str
    userPhoneNumber: str


class CallInitiateResponse(BaseModel):
    callId: str
    twilioCallSid: str
    status: str


class CallEventRequest(BaseModel):
    CallSid: str
    CallStatus: str
    From: str
    To: str
    Direction: Optional[str] = None
    CallDuration: Optional[str] = None
    RecordingUrl: Optional[str] = None
    RecordingSid: Optional[str] = None


class CallEventResponse(BaseModel):
    success: bool
    message: str
    twiml: Optional[str] = None


# Agent management schemas
class AgentInfo(BaseModel):
    agent_id: str
    capabilities: List[str]
    status: str
    current_call: Optional[str] = None


class AgentRegistrationRequest(BaseModel):
    capabilities: Optional[List[str]] = None


class AgentRegistrationResponse(BaseModel):
    success: bool
    message: str


class AgentListResponse(BaseModel):
    agents: Dict[str, Dict[str, Any]]
    call_assignments: Dict[str, str]
    total_agents: int
    busy_agents: int
