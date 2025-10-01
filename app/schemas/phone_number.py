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
    
    @validator('assistant_id', pre=True)
    def validate_assistant_id(cls, v):
        # Convert empty string to None
        if v == "" or v is None:
            return None
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
    twilio_phone_number_sid: Optional[str] = None
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
    
    @validator('assistant_id', pre=True)
    def validate_assistant_id(cls, v):
        # Convert empty string to None
        if v == "" or v is None:
            return None
        return v

class CreatePhoneNumberResponse(BaseModel):
    id: uuid.UUID
    phone_number: str
    label: Optional[str]
    status: str
    created_at: datetime
    message: str
