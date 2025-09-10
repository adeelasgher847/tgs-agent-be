from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid


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
    userPhoneNumber: str


class CallInitiateResponse(BaseModel):
    callId: str
    twilioCallSid: str
    callSessionId: str
    status: str