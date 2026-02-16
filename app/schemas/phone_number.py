from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime
import uuid
import re

class PhoneNumberBase(BaseModel):
    phone_number: str = Field(..., description="Phone number in E.164 format (+1234567890)")
    label: Optional[str] = Field(None, description="Custom label for the phone number")
    status: str = Field(default="active", description="Status: active, inactive")
    
    @validator('phone_number')
    def validate_phone_number(cls, v):
        # Basic E.164 format validation
        if not re.match(r'^\+[1-9]\d{1,14}$', v):
            raise ValueError('Phone number must be in E.164 format (+1234567890)')
        return v

class PhoneNumberCreate(PhoneNumberBase):
    tenant_id: uuid.UUID = Field(..., description="Tenant ID")
    assistant_id: Optional[uuid.UUID] = Field(None, description="Optional assistant to assign to this number")
    
    # Carrier (required for Vicidial - tenant selects which carrier to use for calls)
    carrier_id: Optional[uuid.UUID] = Field(None, description="Carrier ID (required if dialer_type=vicidial)")
    
    # Vicidial fields (optional)
    dialer_type: Optional[str] = Field("twilio", description="Dialer type: twilio or vicidial")
    vicidial_campaign_id: Optional[str] = Field(None, description="Vicidial campaign ID (required if dialer_type=vicidial)")
    caller_id_number: Optional[str] = Field(None, description="Caller ID number for Vicidial")
    
    @validator('assistant_id', pre=True)
    def validate_assistant_id(cls, v):
        # Convert empty string to None
        if v == "" or v is None:
            return None
        return v
    
    @validator('dialer_type')
    def validate_dialer_type(cls, v):
        if v and v not in ['twilio', 'vicidial']:
            raise ValueError('dialer_type must be either "twilio" or "vicidial"')
        return v or 'twilio'
    
    @validator('vicidial_campaign_id')
    def validate_vicidial_campaign(cls, v, values):
        dialer_type = values.get('dialer_type', 'twilio')
        if dialer_type == 'vicidial' and not v:
            raise ValueError('vicidial_campaign_id is required when dialer_type is "vicidial"')
        return v
    
    @validator('carrier_id')
    def validate_carrier_id_for_vicidial(cls, v, values):
        dialer_type = values.get('dialer_type', 'twilio')
        if dialer_type == 'vicidial' and not v:
            raise ValueError('carrier_id is required when dialer_type is "vicidial"')
        return v

class PhoneNumberUpdate(BaseModel):
    label: Optional[str] = None
    status: Optional[str] = None
    assistant_id: Optional[uuid.UUID] = None
    
    @validator('assistant_id', pre=True)
    def validate_assistant_id(cls, v):
        # Convert empty string to None
        if v == "" or v is None:
            return None
        return v

class PhoneNumberResponse(PhoneNumberBase):
    id: uuid.UUID
    tenant_id: uuid.UUID
    assistant_id: Optional[uuid.UUID] = None
    dialer_type: str = Field(default="twilio", description="Dialer type: twilio or vicidial")
    carrier_id: Optional[uuid.UUID] = None
    caller_id_number: Optional[str] = None
    twilio_phone_number_sid: Optional[str] = None
    twilio_account_sid: Optional[str] = None  # Custom Twilio Account SID
    vicidial_cid_group_id: Optional[str] = None
    vicidial_campaign_id: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

class PhoneNumberList(BaseModel):
    phone_numbers: List[PhoneNumberResponse]
    total: int

# Simple request/response models
class CreatePhoneNumberRequest(BaseModel):
    phone_number: str = Field(..., description="Phone number to create")
    label: Optional[str] = Field(None, description="Custom label")
    assistant_id: Optional[uuid.UUID] = Field(None, description="Assistant to assign")
    
    # Carrier (required for Vicidial - tenant selects which carrier to use for calls)
    carrier_id: Optional[uuid.UUID] = Field(None, description="Carrier ID (required if dialer_type=vicidial)")
    
    # Vicidial fields (optional)
    dialer_type: Optional[str] = Field("twilio", description="Dialer type: twilio or vicidial")
    vicidial_campaign_id: Optional[str] = Field(None, description="Vicidial campaign ID (required if dialer_type=vicidial)")
    caller_id_number: Optional[str] = Field(None, description="Caller ID number for Vicidial (user's own number)")
    
    @validator('assistant_id', pre=True)
    def validate_assistant_id(cls, v):
        # Convert empty string to None
        if v == "" or v is None:
            return None
        return v
    
    @validator('dialer_type')
    def validate_dialer_type(cls, v):
        if v and v not in ['twilio', 'vicidial']:
            raise ValueError('dialer_type must be either "twilio" or "vicidial"')
        return v or 'twilio'
    
    @validator('vicidial_campaign_id')
    def validate_vicidial_campaign(cls, v, values):
        dialer_type = values.get('dialer_type', 'twilio')
        if dialer_type == 'vicidial' and not v:
            raise ValueError('vicidial_campaign_id is required when dialer_type is "vicidial"')
        return v
    
    @validator('carrier_id')
    def validate_carrier_id_for_vicidial(cls, v, values):
        dialer_type = values.get('dialer_type', 'twilio')
        if dialer_type == 'vicidial' and not v:
            raise ValueError('carrier_id is required when dialer_type is "vicidial"')
        return v

class CreatePhoneNumberResponse(BaseModel):
    id: uuid.UUID
    phone_number: str
    label: Optional[str]
    status: str
    created_at: datetime
    message: str

class ImportTwilioPhoneNumberRequest(BaseModel):
    """Request schema for importing Twilio phone number"""
    phone_number: str = Field(..., description="Phone number in E.164 format (+1234567890)")
    label: Optional[str] = Field(None, description="Custom label for the phone number")
    twilio_account_sid: str = Field(..., description="Twilio Account SID")
    twilio_auth_token: str = Field(..., description="Twilio Auth Token")
    
    @validator('phone_number')
    def validate_phone_number(cls, v):
        if not re.match(r'^\+[1-9]\d{1,14}$', v):
            raise ValueError('Phone number must be in E.164 format (+1234567890)')
        return v

class ImportTwilioPhoneNumberResponse(BaseModel):
    """Response schema for imported Twilio phone number"""
    id: uuid.UUID
    phone_number: str
    label: Optional[str]
    status: str
    twilio_account_sid: str
    created_at: datetime
    message: str

class VicidialPhoneNumberRequest(BaseModel):
    """Request schema for adding Vicidial phone number"""
    phone_number: str = Field(..., description="Phone number in E.164 format (+1234567890)")
    label: Optional[str] = Field(None, description="Custom label for the phone number")
    carrier_id: uuid.UUID = Field(..., description="Carrier ID (must exist)")
    caller_id_number: str = Field(..., description="Caller ID number (user's own number)")
    vicidial_campaign_id: str = Field(..., description="Vicidial campaign ID")
    assistant_id: Optional[uuid.UUID] = Field(None, description="Optional assistant to assign")
    
    @validator('phone_number')
    def validate_phone_number(cls, v):
        if not re.match(r'^\+[1-9]\d{1,14}$', v):
            raise ValueError('Phone number must be in E.164 format (+1234567890)')
        return v
    
    @validator('caller_id_number')
    def validate_caller_id(cls, v):
        if not re.match(r'^\+[1-9]\d{1,14}$', v):
            raise ValueError('Caller ID number must be in E.164 format (+1234567890)')
        return v

class VicidialPhoneNumberResponse(BaseModel):
    """Response schema for Vicidial phone number"""
    id: uuid.UUID
    phone_number: str
    label: Optional[str]
    status: str
    dialer_type: str
    carrier_id: uuid.UUID
    caller_id_number: str
    vicidial_cid_group_id: Optional[str]
    vicidial_campaign_id: str
    created_at: datetime
    message: str
