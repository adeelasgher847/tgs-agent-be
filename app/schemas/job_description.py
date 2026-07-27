from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field, model_validator


class EmploymentTypeEnum(str, Enum):
    full_time = "full-time"
    contract = "contract"
    remote = "remote"
    hybrid = "hybrid"


class ProcessingStatusEnum(str, Enum):
    pending = "PENDING"
    processing = "PROCESSING"
    ready = "READY"
    failed = "FAILED"


class ExtractedSkill(BaseModel):
    skill: str = Field(..., min_length=1, max_length=120)
    confidence: float = Field(..., ge=0.0, le=1.0)


class JobDescriptionBase(BaseModel):
    job_title: str = Field(..., min_length=1, max_length=255)
    required_skills: List[str] = Field(default_factory=list)
    years_experience_min: int | None = Field(None, ge=0, le=60)
    years_experience_max: int | None = Field(None, ge=0, le=60)
    education_requirements: str | None = None
    location: str | None = Field(None, max_length=255)
    salary_min: Decimal | None = Field(None, ge=0)
    salary_max: Decimal | None = Field(None, ge=0)
    currency: str | None = Field(None, min_length=3, max_length=12)
    employment_type: EmploymentTypeEnum | None = None
    key_responsibilities: List[str] = Field(default_factory=list)
    required_certifications: List[str] = Field(default_factory=list)
    pass_match_threshold: float = Field(
        default=50.0,
        ge=1.0,
        le=100.0,
        description="Minimum overall match percentage (1-100) required for a resume to pass this job.",
    )

    @model_validator(mode="after")
    def validate_salary_range(self) -> "JobDescriptionBase":
        if self.salary_min is not None and self.salary_max is not None and self.salary_min > self.salary_max:
            raise ValueError("salary_min cannot be greater than salary_max")
        if (
            self.years_experience_min is not None
            and self.years_experience_max is not None
            and self.years_experience_min > self.years_experience_max
        ):
            raise ValueError("years_experience_min cannot be greater than years_experience_max")
        return self


class JobDescriptionCreateManual(JobDescriptionBase):
    raw_text: str | None = None


class JobDescriptionUpdate(BaseModel):
    job_title: str | None = Field(None, min_length=1, max_length=255)
    required_skills: List[str] | None = None
    years_experience_min: int | None = Field(None, ge=0, le=60)
    years_experience_max: int | None = Field(None, ge=0, le=60)
    education_requirements: str | None = None
    location: str | None = Field(None, max_length=255)
    salary_min: Decimal | None = Field(None, ge=0)
    salary_max: Decimal | None = Field(None, ge=0)
    currency: str | None = Field(None, min_length=3, max_length=12)
    employment_type: EmploymentTypeEnum | None = None
    key_responsibilities: List[str] | None = None
    required_certifications: List[str] | None = None
    raw_text: str | None = None
    extracted_skills: List[ExtractedSkill] | None = None
    keywords: List[str] | None = None
    skill_weight_matrix: Dict[str, float] | None = None
    matching_criteria: Dict[str, Any] | None = None
    processing_status: ProcessingStatusEnum | None = None
    pass_match_threshold: float | None = Field(
        default=None,
        ge=1.0,
        le=100.0,
        description="Minimum overall match percentage (1-100) required for a resume to pass this job.",
    )

    @model_validator(mode="after")
    def validate_salary_range(self) -> "JobDescriptionUpdate":
        if self.salary_min is not None and self.salary_max is not None and self.salary_min > self.salary_max:
            raise ValueError("salary_min cannot be greater than salary_max")
        if (
            self.years_experience_min is not None
            and self.years_experience_max is not None
            and self.years_experience_min > self.years_experience_max
        ):
            raise ValueError("years_experience_min cannot be greater than years_experience_max")
        return self


class JobDescriptionOut(JobDescriptionBase):
    id: uuid.UUID
    tenant_id: uuid.UUID
    raw_text: str | None = None
    extracted_skills: List[ExtractedSkill] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    skill_weight_matrix: Dict[str, float] = Field(default_factory=dict)
    matching_criteria: Dict[str, Any] = Field(default_factory=dict)
    processing_status: ProcessingStatusEnum
    version: int
    created_at: datetime
    updated_at: datetime | None = None
    created_by: uuid.UUID
    updated_by: uuid.UUID

    class Config:
        from_attributes = True


class JobDescriptionListOut(BaseModel):
    id: uuid.UUID
    job_title: str
    required_skills: List[str] = Field(default_factory=list)
    years_experience_min: int | None = None
    years_experience_max: int | None = None
    education_requirements: str | None = None
    location: str | None = None
    salary_min: Decimal | None = None
    salary_max: Decimal | None = None
    currency: str | None = None
    employment_type: EmploymentTypeEnum | None = None
    key_responsibilities: List[str] = Field(default_factory=list)
    required_certifications: List[str] = Field(default_factory=list)
    matching_criteria: Dict[str, Any] = Field(default_factory=dict)
    raw_text: str | None = None
    pass_match_threshold: float = 50.0
    created_at: datetime

    class Config:
        from_attributes = True
