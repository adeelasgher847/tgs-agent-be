from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Any


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


# Voice call initiation schemas
class CallInitiateRequest(BaseModel):
    agentId: str
    toNumber: str
    # Explicit caller ID — if provided, must match the agent's bound active phone number.
    fromNumber: Optional[str] = None
    phone_number_id: Optional[str] = None  # Optional, user ka selected phone number ID (VAPI style)
    jd_context: Optional[Dict[str, Any]] = None  # Optional JD payload from scheduler/n8n
    # Optional: resolve resume + job from DB and enrich the agent prompt (same as jd_context.jd_id / resume_id)
    jd_id: Optional[str] = None
    resume_id: Optional[str] = None
    appointment_id: Optional[str] = None  # Follow-up reminder: n8n / Trello → initiate
    tenant_id: Optional[str] = None  # Required when using webhook secret (n8n)
    user_id: Optional[str] = None  # Optional, for n8n webhook calls

    # Legacy Monday.com fields (for backward compatibility)
    board_id: Optional[str] = None  # Optional, Monday.com board ID from n8n workflow (legacy)
    monday_item_id: Optional[str] = None  # Optional, Monday.com item ID from n8n workflow (legacy)
    status_column_id: Optional[str] = None  # Optional, Monday.com status column ID from n8n workflow (legacy)
    call_session_id_column_id: Optional[str] = None  # Optional, Monday.com call_session_id column ID from n8n workflow (legacy)

    # Generic CRM fields (for multi-CRM support)
    crm_container_id: Optional[str] = None  # Generic: board_id/list_id/project_id from n8n workflow
    crm_item_id: Optional[str] = None  # Generic: item_id/task_id/issue_id/card_id from n8n workflow
    status_field_id: Optional[str] = None  # Generic: status column/field ID from n8n workflow
    call_session_id_field_id: Optional[str] = None  # Generic: call_session_id field ID from n8n workflow
    crm_type: Optional[str] = None  # "monday" | "clickup" | "jira" | "trello" from n8n workflow
    callFlowId: Optional[str] = None  # Optional UUID — binds this call session to a CallFlow
    # Batch outbound calls — worker passes substituted prompt + record link
    batch_call_record_id: Optional[str] = None
    batch_prompt_override: Optional[str] = None

    @field_validator("toNumber")
    @classmethod
    def validate_to_number(cls, v: str) -> str:
        from app.schemas.phone_number import _validate_e164

        try:
            return _validate_e164(v.strip())
        except ValueError:
            raise ValueError(
                f"Invalid destination phone number '{v}'. Must be E.164 format (e.g. +15551234567)"
            )

    @field_validator("fromNumber")
    @classmethod
    def validate_from_number(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        from app.schemas.phone_number import _validate_e164

        try:
            return _validate_e164(v.strip())
        except ValueError:
            raise ValueError(
                f"Invalid fromNumber '{v}'. Must be E.164 format (e.g. +15550000000)"
            )


class CallInitiateResponse(BaseModel):
    callId: str
    twilioCallSid: str
    callSessionId: str
    status: str
    
    # Legacy Monday.com fields (for backward compatibility)
    board_id: Optional[str] = None  # Echo back Monday.com board ID if provided
    monday_item_id: Optional[str] = None  # Echo back Monday.com item ID if provided
    status_column_id: Optional[str] = None  # Echo back Monday.com status column ID if provided
    call_session_id_column_id: Optional[str] = None  # Echo back Monday.com call_session_id column ID if provided
    
    # Generic CRM fields (for multi-CRM support)
    crm_container_id: Optional[str] = None  # Echo back generic container ID if provided
    crm_item_id: Optional[str] = None  # Echo back generic item ID if provided
    status_field_id: Optional[str] = None  # Echo back generic status field ID if provided
    call_session_id_field_id: Optional[str] = None  # Echo back generic call_session_id field ID if provided
    crm_type: Optional[str] = None  # Echo back CRM type if provided


class CallInitiateErrorResponse(BaseModel):
    """Error response with CRM metadata for n8n workflow"""
    detail: str
    
    # Legacy Monday.com fields (for backward compatibility)
    board_id: Optional[str] = None  # Echo back Monday.com board ID if provided
    monday_item_id: Optional[str] = None  # Echo back Monday.com item ID if provided
    status_column_id: Optional[str] = None  # Echo back Monday.com status column ID if provided
    call_session_id_column_id: Optional[str] = None  # Echo back Monday.com call_session_id column ID if provided
    
    # Generic CRM fields (for multi-CRM support)
    crm_container_id: Optional[str] = None  # Echo back generic container ID if provided
    crm_item_id: Optional[str] = None  # Echo back generic item ID if provided
    status_field_id: Optional[str] = None  # Echo back generic status field ID if provided
    call_session_id_field_id: Optional[str] = None  # Echo back generic call_session_id field ID if provided
    crm_type: Optional[str] = None  # Echo back CRM type if provided
    call_session_id_column_id: Optional[str] = None  # Echo back Monday.com call_session_id column ID if provided


# Web-based voice chat schemas (Talk to Assistant feature)
class VoiceChatStartRequest(BaseModel):
    agent_id: str


class VoiceChatStartResponse(BaseModel):
    session_id: str
    agent_name: str
    agent_voice_type: Optional[str] = None
    status: str


class VoiceChatMessageRequest(BaseModel):
    session_id: str
    message: str
    message_type: str = "text"  # "text" or "speech"


class VoiceChatMessageResponse(BaseModel):
    session_id: str
    agent_response: str
    response_time: float
    audio_url: Optional[str] = None  # URL to generated speech audio
    timestamp: str


class VoiceChatHistoryRequest(BaseModel):
    session_id: str


class VoiceChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]
    agent_info: Dict[str, Any]


class VoiceChatEndRequest(BaseModel):
    session_id: str


class VoiceChatEndResponse(BaseModel):
    session_id: str
    status: str
    duration: Optional[float] = None
    message_count: int


# Live Voice Conversation schemas (Talk to Assistant feature)
class LiveVoiceStartRequest(BaseModel):
    agent_id: str


class LiveVoiceStartResponse(BaseModel):
    session_id: str
    agent_name: str
    agent_voice_type: Optional[str] = None
    status: str


class LiveVoiceMessageRequest(BaseModel):
    session_id: str
    message: str
    message_type: str = "speech"  # "speech" or "text"


class LiveVoiceMessageResponse(BaseModel):
    session_id: str
    agent_response: str
    response_time: float
    timestamp: str
    should_speak: bool = True