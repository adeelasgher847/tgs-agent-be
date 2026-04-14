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
    years_experience_min: Optional[int] = Field(None, ge=0, le=60)
    years_experience_max: Optional[int] = Field(None, ge=0, le=60)
    education_requirements: Optional[str] = None
    location: Optional[str] = Field(None, max_length=255)
    salary_min: Optional[Decimal] = Field(None, ge=0)
    salary_max: Optional[Decimal] = Field(None, ge=0)
    currency: Optional[str] = Field(None, min_length=3, max_length=12)
    employment_type: Optional[EmploymentTypeEnum] = None
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
    raw_text: Optional[str] = None


class JobDescriptionUpdate(BaseModel):
    job_title: Optional[str] = Field(None, min_length=1, max_length=255)
    required_skills: Optional[List[str]] = None
    years_experience_min: Optional[int] = Field(None, ge=0, le=60)
    years_experience_max: Optional[int] = Field(None, ge=0, le=60)
    education_requirements: Optional[str] = None
    location: Optional[str] = Field(None, max_length=255)
    salary_min: Optional[Decimal] = Field(None, ge=0)
    salary_max: Optional[Decimal] = Field(None, ge=0)
    currency: Optional[str] = Field(None, min_length=3, max_length=12)
    employment_type: Optional[EmploymentTypeEnum] = None
    key_responsibilities: Optional[List[str]] = None
    required_certifications: Optional[List[str]] = None
    raw_text: Optional[str] = None
    extracted_skills: Optional[List[ExtractedSkill]] = None
    keywords: Optional[List[str]] = None
    skill_weight_matrix: Optional[Dict[str, float]] = None
    matching_criteria: Optional[Dict[str, Any]] = None
    processing_status: Optional[ProcessingStatusEnum] = None
    pass_match_threshold: Optional[float] = Field(
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
    raw_text: Optional[str] = None
    extracted_skills: List[ExtractedSkill] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    skill_weight_matrix: Dict[str, float] = Field(default_factory=dict)
    matching_criteria: Dict[str, Any] = Field(default_factory=dict)
    processing_status: ProcessingStatusEnum
    version: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    created_by: uuid.UUID
    updated_by: uuid.UUID

    class Config:
        from_attributes = True


class JobDescriptionListOut(BaseModel):
    id: uuid.UUID
    job_title: str
    required_skills: List[str] = Field(default_factory=list)
    years_experience_min: Optional[int] = None
    years_experience_max: Optional[int] = None
    education_requirements: Optional[str] = None
    location: Optional[str] = None
    salary_min: Optional[Decimal] = None
    salary_max: Optional[Decimal] = None
    currency: Optional[str] = None
    employment_type: Optional[EmploymentTypeEnum] = None
    key_responsibilities: List[str] = Field(default_factory=list)
    required_certifications: List[str] = Field(default_factory=list)
    raw_text: Optional[str] = None
    pass_match_threshold: float = 50.0
    created_at: datetime

    class Config:
        from_attributes = True
