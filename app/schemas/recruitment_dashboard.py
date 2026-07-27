from __future__ import annotations

from datetime import datetime
from typing import List, Optional
import uuid

from pydantic import BaseModel, Field


class RecruitmentFunnelRow(BaseModel):
    """One stage in the candidate pipeline (matches the funnel chart in the UI)."""

    key: str = Field(
        description="Machine key: sourcing, screened, technical, manager, offer",
    )
    label: str
    count: int = Field(ge=0)


class UpcomingInterviewItem(BaseModel):
    resume_id: uuid.UUID
    interview_id: uuid.UUID
    candidate_name: str
    candidate_initials: str
    job_title: Optional[str] = None
    scheduled_at: datetime
    time_label: str = Field(
        description='Human label such as "Today 3:00 PM" in UTC calendar'
    )
    is_today: bool
    is_tomorrow: bool


class ActiveJobRow(BaseModel):
    job_description_id: uuid.UUID
    job_title: str
    open_roles: int = Field(
        description="Planned or configured openings (matching_criteria.open_roles/headcount, default 1)"
    )
    applicant_count: int = Field(
        description="Resumes currently associated with this job",
    )
    posted_at: datetime
    posted_ago: str


class AccountSnapshot(BaseModel):
    user_id: uuid.UUID
    email: str
    tenant_id: uuid.UUID
    tenant_name: str
    credits: float
    currency_label: str = "credits"


class RecruitmentKpiBlock(BaseModel):
    open_positions: int
    open_positions_subtitle: Optional[str] = None
    total_candidates: int
    total_candidates_subtitle: Optional[str] = None
    interviews_scheduled: int
    interviews_today: int
    interviews_scheduled_subtitle: Optional[str] = None
    offers_sent: int
    offers_awaiting_feedback: int
    offers_subtitle: Optional[str] = None


class RecruitmentDashboardData(BaseModel):
    summary: RecruitmentKpiBlock
    pipeline: List[RecruitmentFunnelRow]
    upcoming_interviews: List[UpcomingInterviewItem]
    active_openings: List[ActiveJobRow]
    account: AccountSnapshot
