from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ResumeInterviewScheduleRequest(BaseModel):
    resume_id: UUID
    crm_config_id: UUID | None = Field(
        default=None,
        description="Optional CRM config ID. If omitted, backend auto-selects user's Trello-linked CRM (fallback: first linked CRM).",
    )
    agent_id: UUID
    phone_number: str = Field(description="Candidate phone number in E.164 format")
    call_time_utc: str = Field(description="Scheduled UTC datetime string")
    job_description_id: UUID | None = None
    phone_number_id: UUID | None = None
    metadata: dict[str, Any] | None = None


class ResumeInterviewBulkScheduleRequest(BaseModel):
    items: list[ResumeInterviewScheduleRequest] = Field(min_length=1, max_length=500)


class ResumeInterviewBulkScheduleResultItem(BaseModel):
    resume_id: UUID
    success: bool
    interview: ResumeInterviewItem | None = None
    error: str | None = None


class ResumeInterviewBulkScheduleResponse(BaseModel):
    total: int
    success_count: int
    error_count: int
    items: list[ResumeInterviewBulkScheduleResultItem] = Field(default_factory=list)


class ResumeInterviewStatusUpdateRequest(BaseModel):
    status: str
    call_session_id: UUID | None = None
    twilio_call_sid: str | None = None
    last_error: str | None = None
    increment_attempt: bool = False
    metadata_patch: dict[str, Any] | None = None


class ResumeInterviewItem(BaseModel):
    id: UUID
    tenant_id: UUID
    resume_id: UUID
    job_description_id: UUID | None = None
    agent_id: UUID
    call_session_id: UUID | None = None
    candidate_phone: str
    scheduled_at: datetime
    status: str
    crm_type: str | None = None
    crm_item_id: str | None = None
    crm_batch_id: str | None = None
    phone_number_id: UUID | None = None
    twilio_call_sid: str | None = None
    attempt_count: int
    last_error: str | None = None
    metadata_json: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime | None = None


class ResumeInterviewCalendarItem(BaseModel):
    interview_id: UUID
    resume_id: UUID
    resume_filename: str
    scheduled_at: datetime
    status: str
    agent_id: UUID
    candidate_phone: str
    candidate_name: str | None = Field(
        default=None,
        description="Candidate full name from resume parsed_json.profile.name when available.",
    )
    candidate_email: str | None = Field(
        default=None,
        description="Candidate email from resume parsed_json.profile.email when available.",
    )
    job_description_id: UUID | None = None
    call_session_id: UUID | None = None
    transcript: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Call transcript for linked call session, if available.",
    )


class ResumeInterviewSessionLinkItem(BaseModel):
    resume_id: UUID
    resume_filename: str
    interview_id: UUID | None = None
    interview_status: str | None = None
    scheduled_at: datetime | None = None
    call_session_id: UUID | None = None
    call_session_status: str | None = None
    twilio_call_sid: str | None = None
    crm_item_id: str | None = None
    crm_batch_id: str | None = None


class ResumeInterviewCallMediaResponse(BaseModel):
    """Call transcript and recording for the latest resume interview linked to a resume."""

    resume_id: UUID
    interview_id: UUID
    call_session_id: UUID
    recording_url: str | None = Field(
        default=None,
        description="Voice recording URL on the linked CallSession when the provider stored one.",
    )
    twilio_call_sid: str | None = None
    call_session_status: str | None = None
    transcript: list[dict[str, Any]] = Field(
        default_factory=list,
        description='Conversation turns with at least "role" and "content" (from TranscriptMessage rows if any, else call_transcript JSON).',
    )
    transcript_source: str = Field(
        ...,
        description='One of: "transcript_messages", "call_session", "empty".',
    )


class ResumeInterviewTrelloCallMediaResponse(BaseModel):
    """Call media resolved from Trello card via resume interview id."""

    resume_interview_id: UUID
    resume_id: UUID
    trello_card_id: str
    call_session_id: UUID
    recording_url: str | None = None
    twilio_call_sid: str | None = None
    call_session_status: str | None = None
    transcript: list[dict[str, Any]] = Field(
        default_factory=list,
        description='Conversation turns with at least "role" and "content".',
    )
    transcript_source: str = Field(
        ...,
        description='One of: "transcript_messages", "call_session", "empty".',
    )

