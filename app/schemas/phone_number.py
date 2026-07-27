"""
Phone number schemas — Pydantic v2.

Naming note: the DB column is `assistant_id` (legacy); public API uses `agent_id` only.
The service layer maps agent_id ↔ assistant_id on the ORM model.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

E164_RE = re.compile(r"^\+[1-9]\d{1,14}$")

DAYS_OF_WEEK = frozenset(range(7))  # 0 = Monday … 6 = Sunday
TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def _validate_e164(v: str) -> str:
    if not E164_RE.match(v):
        raise ValueError("Phone number must be in E.164 format (e.g. +614xxxxxxxx)")
    return v


# ---------------------------------------------------------------------------
# Legacy CRUD schemas (kept for backward compat with existing router)
# ---------------------------------------------------------------------------


class PhoneNumberBase(BaseModel):
    phone_number: str = Field(..., description="Phone number in E.164 format")
    label: Optional[str] = Field(None)
    status: str = Field(default="active")

    @field_validator("phone_number")
    @classmethod
    def validate_phone_number(cls, v: str) -> str:
        return _validate_e164(v)


class PhoneNumberCreate(PhoneNumberBase):
    """Internal create payload; maps agent_id to DB column assistant_id."""

    tenant_id: uuid.UUID
    assistant_id: Optional[uuid.UUID] = None

    @field_validator("assistant_id", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: Any) -> Any:
        return None if v == "" else v


class PhoneNumberUpdate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    label: Optional[str] = None
    status: Optional[str] = None
    agent_id: Optional[uuid.UUID] = Field(
        default=None,
        validation_alias=AliasChoices("agent_id", "agentId"),
        description="Agent to bind (stored as assistant_id in DB)",
    )

    @field_validator("agent_id", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: Any) -> Any:
        return None if v == "" else v


class PhoneNumberResponse(PhoneNumberBase):
    """Phone number detail — never includes twilio_auth_token, sip_password, or other secrets."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    provider: str = "twilio"
    agent_id: Optional[uuid.UUID] = Field(
        default=None,
        validation_alias=AliasChoices("assistant_id", "agent_id", "agentId"),
        description="Bound agent id (maps from DB column assistant_id)",
    )
    twilio_phone_number_sid: Optional[str] = None
    twilio_account_sid: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class PhoneNumberList(BaseModel):
    phone_numbers: List[PhoneNumberResponse]
    total: int


class CreatePhoneNumberRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    phone_number: str
    label: Optional[str] = None
    agent_id: Optional[uuid.UUID] = Field(
        default=None,
        validation_alias=AliasChoices("agent_id", "agentId"),
        description="Optional agent to bind on create",
    )

    @field_validator("phone_number")
    @classmethod
    def validate_phone_number(cls, v: str) -> str:
        return _validate_e164(v)

    @field_validator("agent_id", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: Any) -> Any:
        return None if v == "" else v


class CreatePhoneNumberResponse(BaseModel):
    id: uuid.UUID
    phone_number: str
    label: Optional[str]
    status: str
    created_at: datetime
    message: str


class ImportTwilioPhoneNumberRequest(BaseModel):
    phone_number: str = Field(..., description="E.164 phone number")
    label: Optional[str] = None
    twilio_account_sid: str
    twilio_auth_token: str

    @field_validator("phone_number")
    @classmethod
    def validate_phone_number(cls, v: str) -> str:
        return _validate_e164(v)


class ImportTwilioPhoneNumberResponse(BaseModel):
    id: uuid.UUID
    phone_number: str
    label: Optional[str]
    status: str
    twilio_account_sid: str
    created_at: datetime
    message: str


# ---------------------------------------------------------------------------
# Number configuration (recording, duration, business hours)
# ---------------------------------------------------------------------------


class ScheduleEntry(BaseModel):
    day: int = Field(..., ge=0, le=6, description="0=Monday … 6=Sunday")
    open: str = Field(..., description="HH:mm open time")
    close: str = Field(..., description="HH:mm close time")

    @field_validator("open", "close")
    @classmethod
    def validate_time(cls, v: str) -> str:
        if not TIME_RE.match(v):
            raise ValueError("Time must be in HH:mm format (e.g. 09:00)")
        return v


class BusinessHoursSchema(BaseModel):
    timezone: str = Field(..., description="IANA timezone e.g. Australia/Sydney")
    schedule: List[ScheduleEntry]


class NumberConfigurationRequest(BaseModel):
    recording_enabled: bool = False
    max_duration_seconds: int = Field(default=3600, ge=60, le=86400)
    business_hours: Optional[BusinessHoursSchema] = None


class NumberConfigurationResponse(BaseModel):
    id: uuid.UUID
    phone_number_id: uuid.UUID
    recording_enabled: bool
    max_duration_seconds: int
    business_hours: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Sprint 2 — new provisioning API schemas
# ---------------------------------------------------------------------------


class PhoneNumberSearchResult(BaseModel):
    phone_number: str
    friendly_name: Optional[str] = None
    locality: Optional[str] = None
    region: Optional[str] = None
    country: str
    capabilities: Dict[str, bool]
    beta: bool = False


class PhoneNumberSearchResponse(BaseModel):
    available_numbers: List[PhoneNumberSearchResult]
    total: int


class PurchasePhoneNumberRequest(BaseModel):
    """POST /api/v1/phone-numbers/purchase"""

    phone_number: str = Field(..., description="E.164 number to purchase e.g. +61412345678")
    label: Optional[str] = None

    @field_validator("phone_number")
    @classmethod
    def validate_phone_number(cls, v: str) -> str:
        return _validate_e164(v)


class PurchasePhoneNumberResponse(BaseModel):
    id: uuid.UUID
    phone_number: str
    provider: str
    twilio_sid: Optional[str]
    status: str
    workspace_id: uuid.UUID
    created_at: datetime
    message: str


# ---------------------------------------------------------------------------
# External / BYO number registration
# ---------------------------------------------------------------------------


class RegisterExternalNumberRequest(BaseModel):
    """POST /api/v1/telephony/external"""

    phone_number: str = Field(..., description="E.164 number e.g. +61412345678")
    label: Optional[str] = None
    sip_username: str = Field(..., description="SIP username for inbound routing")
    sip_password: str = Field(..., description="SIP password (stored encrypted)")

    @field_validator("phone_number")
    @classmethod
    def validate_phone_number(cls, v: str) -> str:
        return _validate_e164(v)


class RegisterExternalNumberResponse(BaseModel):
    """External number registration result — sip_password is write-only and never returned."""

    id: uuid.UUID
    phone_number: str
    provider: Literal["external"]
    status: str
    workspace_id: uuid.UUID
    created_at: datetime
    message: str


# ---------------------------------------------------------------------------
# Bind / Unbind
# ---------------------------------------------------------------------------


class BindNumberRequest(BaseModel):
    """POST /api/v1/telephony/bind"""

    model_config = ConfigDict(populate_by_name=True)

    number_id: uuid.UUID = Field(
        ...,
        description="phone_numbers.id",
        validation_alias=AliasChoices("number_id", "numberId"),
    )
    agent_id: uuid.UUID = Field(
        ...,
        description="agent.id to bind (stored as phonenumber.assistant_id)",
        validation_alias=AliasChoices("agent_id", "agentId"),
    )


class UnbindNumberRequest(BaseModel):
    """POST /api/v1/telephony/unbind"""

    model_config = ConfigDict(populate_by_name=True)

    number_id: uuid.UUID = Field(
        ...,
        description="phone_numbers.id",
        validation_alias=AliasChoices("number_id", "numberId"),
    )


class BindingStatusResponse(BaseModel):
    number_id: uuid.UUID
    phone_number: str
    agent_id: Optional[uuid.UUID]
    agent_name: Optional[str]
    agent_status: Optional[str]
    message: str


class BoundAgentBinding(BaseModel):
    """GET /api/v1/telephony/bindings — one row per bound number ↔ agent."""

    agent_id: uuid.UUID
    agent_name: Optional[str] = None
    agent_status: Optional[str] = None
    number_id: uuid.UUID
    phone_number: str


class BoundAgentBindingList(BaseModel):
    bindings: List[BoundAgentBinding]
    total: int


# ---------------------------------------------------------------------------
# Extended list response (binding status + agent name)
# ---------------------------------------------------------------------------


class PhoneNumberWithBinding(BaseModel):
    id: uuid.UUID
    phone_number: str
    provider: str
    label: Optional[str]
    status: str
    workspace_id: uuid.UUID
    twilio_sid: Optional[str]
    binding_status: str  # bound | unbound
    agent_id: Optional[uuid.UUID]
    agent_name: Optional[str]
    agent_status: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class PhoneNumberWithBindingList(BaseModel):
    phone_numbers: List[PhoneNumberWithBinding]
    total: int
