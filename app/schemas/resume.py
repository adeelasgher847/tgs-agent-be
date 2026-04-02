from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ParseStatusEnum(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"


class ParseSource(str, Enum):
    RULES = "RULES"
    AI = "AI"
    HYBRID = "HYBRID"


class ParseMode(str, Enum):
    rules = "rules"
    llm = "llm"
    hybrid = "hybrid"


class SkillItem(BaseModel):
    name: str
    source: str = Field(description="RULES | AI | HYBRID")
    confidence: float = Field(ge=0.0, le=1.0)


class ExperienceItem(BaseModel):
    role: str | None = None
    company: str | None = None
    duration: str | None = None
    responsibilities: list[str] = Field(default_factory=list)


class EducationItem(BaseModel):
    degree: str | None = None
    institution: str | None = None
    year: int | str | None = None


class CertificationItem(BaseModel):
    name: str
    issuer: str | None = None
    year: int | str | None = None


class ProjectItem(BaseModel):
    name: str
    description: str | None = None
    technologies: list[str] = Field(default_factory=list)


class ProfileBlock(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    links: list[str] = Field(default_factory=list)


class ParsedResume(BaseModel):
    profile: ProfileBlock = Field(default_factory=ProfileBlock)
    skills: list[SkillItem] = Field(default_factory=list)
    experience: list[ExperienceItem] = Field(default_factory=list)
    education: list[EducationItem] = Field(default_factory=list)
    certifications: list[CertificationItem] = Field(default_factory=list)
    projects: list[ProjectItem] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    years_experience_total: float | None = None
    raw_text: str | None = None
    parse_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    parse_source: ParseSource = ParseSource.RULES
    parser_version: str = ""
    model_name: str | None = None
    provider: str | None = None


class ResumeStatusResponse(BaseModel):
    resume_id: UUID
    status: ParseStatusEnum
    parse_confidence: float | None = None
    parse_source: str | None = None
    warnings: list[str] = Field(default_factory=list)
    error_message: str | None = None


class ResumeListItem(BaseModel):
    id: UUID
    original_filename: str
    status: ParseStatusEnum
    parse_confidence: float | None = None
    created_at: Any
    upload_mode: str | None = None
    batch_id: UUID | None = None


class BatchShortlistItem(BaseModel):
    resume_id: UUID
    filename: str = Field(description="Original upload filename (e.g. PDF name)")
    score: float = Field(ge=0.0, le=1.0, description="Overall match score vs the job description")


class BatchShortlistPayload(BaseModel):
    items: list[BatchShortlistItem] = Field(
        description="Ranked best-first; length respects top_k / min_overall_score filters",
    )
    not_scored_count: int = Field(
        ge=0,
        description="Resumes in the batch that were skipped (not READY, parse missing, or scorer error)",
    )


class MatchComponentScore(BaseModel):
    criterion: str
    weight: float
    score: float
    matched: bool
    detail: str = ""


class MatchResponse(BaseModel):
    resume_id: UUID
    job_description_id: UUID
    overall_score: float
    skill_match_score: float
    criteria_breakdown: list[MatchComponentScore]
    missing_required_skills: list[str] = Field(default_factory=list)
    weighted_skill_hits: dict[str, float] = Field(default_factory=dict)


class MatchRequest(BaseModel):
    parse_mode: ParseMode | None = Field(
        default=None, description="Optional parse mode hint for client workflows"
    )

