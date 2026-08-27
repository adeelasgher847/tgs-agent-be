"""Pydantic schemas for Inbound Rules & Blocklist Rule Sets."""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def normalize_phone_digits(phone: str) -> str:
    """Strip all non-digit characters from phone number string."""
    return re.sub(r"\D", "", phone or "")


class InboundRuleItem(BaseModel):
    id: Optional[uuid.UUID] = None
    phone_number_pattern: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Raw phone number or pattern, e.g. '+1 (555) 987-6543'",
    )
    normalized_digits: Optional[str] = Field(
        None,
        description="Digits-only sanitized string for fast matching",
    )
    label: Optional[str] = Field(
        None,
        max_length=100,
        description="Optional descriptive label, e.g. 'Robocaller' or 'Spam'",
    )
    action: str = Field(
        default="deny",
        description="Rule action, default is 'deny'",
    )
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    @field_validator("phone_number_pattern")
    @classmethod
    def validate_phone_pattern(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("phone_number_pattern cannot be empty or whitespace")
        digits = normalize_phone_digits(s)
        if not digits:
            raise ValueError("phone_number_pattern must contain at least one digit")
        return s

    @field_validator("label")
    @classmethod
    def clean_label(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = v.strip()
        return s if s else None


class InboundRuleSetCreate(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of the rule set, e.g. 'Global Spammers'",
    )
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="Optional description of the rule set",
    )
    rules: List[InboundRuleItem] = Field(
        default_factory=list,
        description="Initial list of number rules to add",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("name")
    @classmethod
    def clean_name(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("name cannot be empty or whitespace")
        return s

    @field_validator("description")
    @classmethod
    def clean_description(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = v.strip()
        return s if s else None


class InboundRuleSetUpdate(BaseModel):
    name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=100,
        description="Updated name of the rule set",
    )
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="Updated description of the rule set",
    )
    rules: Optional[List[InboundRuleItem]] = Field(
        None,
        description="Full replacement list of number rules (if provided)",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("name")
    @classmethod
    def clean_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = v.strip()
        if not s:
            raise ValueError("name cannot be empty or whitespace")
        return s

    @field_validator("description")
    @classmethod
    def clean_description(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = v.strip()
        return s if s else None


class InboundRuleSetListItem(BaseModel):
    id: uuid.UUID
    name: str
    description: Optional[str] = None
    rules_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class InboundRuleSetResponse(BaseModel):
    id: uuid.UUID
    name: str
    description: Optional[str] = None
    rules_count: int = 0
    rules: List[InboundRuleItem] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class InboundRuleImportRequest(BaseModel):
    raw_text: Optional[str] = Field(
        None,
        description="Multiline text containing phone numbers, CSV data, or comma-separated rows with optional labels",
    )
    rule_set_id: Optional[uuid.UUID] = Field(
        None,
        description="Existing rule set ID to append imported rules to",
    )
    new_rule_set_name: Optional[str] = Field(
        None,
        max_length=100,
        description="Name of the new rule set to create if rule_set_id is not provided",
    )
    new_rule_set_description: Optional[str] = Field(
        None,
        max_length=500,
        description="Description of the new rule set if created",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("new_rule_set_name")
    @classmethod
    def clean_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = v.strip()
        return s if s else None


class InboundRuleImportResponse(BaseModel):
    rule_set: InboundRuleSetResponse
    imported_count: int
    skipped_count: int
    total_rules_count: int

    model_config = ConfigDict(from_attributes=True)


class FlowInboundRulesUpdate(BaseModel):
    inbound_rule_set_id: Optional[uuid.UUID] = Field(
        None,
        description="Inbound rule set ID to attach to the flow, or null to detach",
    )

    model_config = ConfigDict(extra="forbid")


class FlowInboundRulesResponse(BaseModel):
    inbound_rule_set_id: Optional[uuid.UUID] = None
    inbound_rule_set_name: Optional[str] = None
    active_rules_count: int = 0

    model_config = ConfigDict(from_attributes=True)
