from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, computed_field

from app.utils.fit_score_labels import explain_fit_score


class ParseStatusEnum(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"


class CandidateStatusEnum(str, Enum):
    QUALIFIED = "qualified"
    PARTIALLY_QUALIFIED = "partially qualified"
    REJECTED = "rejected"


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
    location: str | None = None
    education: list[str] = Field(default_factory=list)
    experience_years: float | None = None
    title: str | None = None
    summary: str | None = None
    phone: str | None = None
    email: str | None = None
    skills: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    match_percent: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Resume vs job match as 0–100 (same scale as match endpoints). Null if not scored.",
    )
    overall_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Raw match strength 0–1 when scored.",
    )
    overall_match_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Alias of overall_score for client compatibility.",
    )
    fit_label: str | None = Field(
        default=None,
        description='Exactly "Relevant" or "Irrelevant" when scored (server threshold on overall score).',
    )
    candidate_status: CandidateStatusEnum | None = Field(
        default=None,
        description='Manual reviewer decision: "qualified", "partially qualified", or "rejected".',
    )
    is_relevant: bool | None = Field(
        default=None,
        description="True = relevant to this job, False = irrelevant, null = not scored.",
    )
    created_at: Any
    batch_id: UUID | None = None
    job_description_id: UUID | None = None
    resume_interviews: list[UUID] = Field(default_factory=list)


class BatchShortlistItem(BaseModel):
    resume_id: UUID
    filename: str = Field(description="Original upload filename (e.g. PDF name)")
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Technical match strength 0–1 (for APIs and sorting)",
    )
    match_percent: int = Field(
        ge=0,
        le=100,
        description="Same as score expressed as a simple 0–100 number",
    )
    fit_label: str = Field(
        description='Either "Relevant" or "Irrelevant" (cutoff uses server RELEVANCE_THRESHOLD in fit_score_labels)',
    )
    fit_summary: str = Field(
        description="One short sentence anyone can understand—no jargon",
    )


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
    parse_confidence: float | None = Field(
        default=None,
        description="Confidence of resume parsing quality (0-1).",
    )
    match_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in this match result quality (0-1), separate from the match score.",
    )
    match_source: str = Field(
        default="rules",
        description="rules | ai | hybrid — how overall_score was produced",
    )
    rules_baseline_overall: float | None = Field(
        default=None,
        description="Deterministic overall score before LLM blend (hybrid/ai modes)",
    )
    ai_rationale: str | None = Field(default=None, description="Short LLM justification when AI ran")
    ai_red_flags: list[str] = Field(
        default_factory=list,
        description="LLM-noted mismatches or risks",
    )
    ai_model: str | None = None
    ai_provider: str | None = None

    @computed_field
    @property
    def overall_match_percent(self) -> int:
        return explain_fit_score(self.overall_score)[0]

    @computed_field
    @property
    def overall_fit_label(self) -> str:
        return explain_fit_score(self.overall_score)[1]

    @computed_field
    @property
    def overall_fit_summary(self) -> str:
        return explain_fit_score(self.overall_score)[2]

    @computed_field
    @property
    def skill_match_percent(self) -> int:
        s = max(0.0, min(1.0, float(self.skill_match_score)))
        return int(round(s * 100))

    @computed_field
    @property
    def match_confidence_percent(self) -> int:
        c = max(0.0, min(1.0, float(self.match_confidence)))
        return int(round(c * 100))

    @computed_field
    @property
    def match_confidence_label(self) -> str:
        c = max(0.0, min(1.0, float(self.match_confidence)))
        if c >= 0.75:
            return "High"
        if c >= 0.5:
            return "Medium"
        return "Low"


class MatchMode(str, Enum):
    """How to combine LLM judgement with deterministic rules."""

    rules = "rules"
    ai = "ai"
    hybrid = "hybrid"


class MatchRequest(BaseModel):
    parse_mode: ParseMode | None = Field(
        default=None, description="Optional parse mode hint for client workflows"
    )
    match_mode: MatchMode | None = Field(
        default=None,
        description="rules = heuristics only; ai = LLM primary (rules on parse fail); hybrid = blend (default from server settings)",
    )


class ShortlistCriteriaUpdateRequest(BaseModel):
    skill_weight_matrix: dict[str, float] | None = Field(
        default=None,
        description="Optional skill weight map. Values are normalized server-side.",
    )
    scoring_dimensions: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional scoring dimensions list with `name`, `weight`, and optional `description`.",
    )
    must_have_criteria: list[str] | None = Field(
        default=None,
        description="Optional list of hard filters (human-readable constraints).",
    )
    minimum_parse_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum parse confidence required for shortlisting.",
    )
    minimum_profile_completeness: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum profile completeness required for shortlisting.",
    )


class ShortlistCriteriaResponse(BaseModel):
    job_description_id: UUID
    matching_criteria: dict[str, Any] = Field(default_factory=dict)
    skill_weight_matrix: dict[str, float] = Field(default_factory=dict)
    version: int


class TopCandidateItem(BaseModel):
    resume_id: UUID
    filename: str
    score: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)
    match_percent: int = Field(ge=0, le=100)
    fit_label: str
    fit_summary: str
    profile_completeness: float = Field(ge=0.0, le=1.0)
    parse_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    exclusion_reasons: list[str] = Field(default_factory=list)


class TopCandidatesResponse(BaseModel):
    items: list[TopCandidateItem] = Field(default_factory=list)
    scanned_count: int = Field(ge=0)
    shortlisted_count: int = Field(ge=0)
    excluded_count: int = Field(ge=0)
    excluded_reasons_summary: dict[str, int] = Field(default_factory=dict)


class TopCandidatesRequest(BaseModel):
    job_description_id: UUID
    batch_id: UUID | None = Field(default=None, description="Optional: shortlist within one uploaded batch.")
    top_k: int = Field(default=20, ge=1, le=200)
    min_overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    max_resumes: int = Field(default=1000, ge=1, le=5000)
    match_mode: MatchMode | None = Field(default=None, description="rules | ai | hybrid")
    include_excluded: bool = Field(
        default=False,
        description="If true, include exclusion reasons on returned records (when applicable).",
    )


class ShortlistByBatchRequest(BaseModel):
    batch_id: UUID = Field(description="Upload batch UUID from resume multi-upload.")
    job_description_id: UUID = Field(description="Job description UUID to match against.")
    top_k: int = Field(default=20, ge=1, le=200)
    min_overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    max_resumes: int = Field(default=1000, ge=1, le=5000)
    match_mode: MatchMode | None = Field(default=None, description="rules | ai | hybrid")
    include_excluded: bool = Field(default=False)


class ResumeCandidateStatusUpdateRequest(BaseModel):
    status: CandidateStatusEnum


class ResumeCandidateStatusUpdateResponse(BaseModel):
    resume_id: UUID
    status: CandidateStatusEnum

