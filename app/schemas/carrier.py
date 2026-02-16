from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
import uuid

class CarrierBase(BaseModel):
    name: str = Field(..., description="Carrier name (e.g., 'Vonage US', 'Telenyx EU')")
    provider: str = Field(..., description="Carrier provider: 'Vonage', 'Telenyx', 'Twilio'")
    status: str = Field(default="active", description="Status: active, inactive")
    description: Optional[str] = Field(None, description="Optional description")
    
    @validator('provider')
    def validate_provider(cls, v):
        allowed_providers = ['Vonage', 'Telenyx', 'Twilio', 'Other']
        if v not in allowed_providers:
            raise ValueError(f'Provider must be one of: {", ".join(allowed_providers)}')
        return v

class CarrierCreate(CarrierBase):
    tenant_id: Optional[uuid.UUID] = Field(default=None, description="Tenant ID (optional - leave null for global carriers)")
    sip_username: Optional[str] = Field(default=None, description="SIP username (will be encrypted)")
    sip_password: Optional[str] = Field(default=None, description="SIP password (will be encrypted)")
    sip_server: Optional[str] = Field(default=None, description="SIP server URL (e.g., sip.vonage.com)")
    sip_port: Optional[int] = Field(default=5060, description="SIP port (default: 5060)")
    vicidial_carrier_id: Optional[str] = Field(default=None, description="Carrier ID in Vicidial")

class CarrierUpdate(BaseModel):
    name: Optional[str] = None
    provider: Optional[str] = None
    status: Optional[str] = None
    description: Optional[str] = None
    sip_username: Optional[str] = None
    sip_password: Optional[str] = None
    sip_server: Optional[str] = None
    sip_port: Optional[int] = None
    vicidial_carrier_id: Optional[str] = None
    
    @validator('provider')
    def validate_provider(cls, v):
        if v is not None:
            allowed_providers = ['Vonage', 'Telenyx', 'Twilio', 'Other']
            if v not in allowed_providers:
                raise ValueError(f'Provider must be one of: {", ".join(allowed_providers)}')
        return v

class CarrierResponse(CarrierBase):
    id: uuid.UUID
    tenant_id: Optional[uuid.UUID] = None  # Nullable for global carriers
    sip_server: Optional[str] = None
    sip_port: Optional[int] = None
    vicidial_carrier_id: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class CarrierList(BaseModel):
    carriers: List[CarrierResponse]
    total: int
