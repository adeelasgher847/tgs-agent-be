from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer, Numeric, Float
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
import uuid

from app.db.base_class import Base


class JobDescription(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id"), nullable=False, index=True)

    # Recruiter-provided fields
    job_title = Column(String(255), nullable=False)
    required_skills = Column(JSONB, nullable=True)  # list[str
    years_experience_min = Column(Integer, nullable=True)
    years_experience_max = Column(Integer, nullable=True)
    education_requirements = Column(Text, nullable=True)
    location = Column(String(255), nullable=True)
    salary_min = Column(Numeric(12, 2), nullable=True)
    salary_max = Column(Numeric(12, 2), nullable=True)
    currency = Column(String(12), nullable=True)
    employment_type = Column(String(50), nullable=True)
    key_responsibilities = Column(JSONB, nullable=True)  # list[str] or free-text wrapped as list
    required_certifications = Column(JSONB, nullable=True)  # list[str]
    pass_match_threshold = Column(Float, nullable=False, default=0.5, server_default="0.5")

    # System/AI fields
    raw_text = Column(Text, nullable=True)
    extracted_skills = Column(JSONB, nullable=True)  # list[{"skill": str, "confidence": float}]
    keywords = Column(JSONB, nullable=True)  # list[str]
    skill_weight_matrix = Column(JSONB, nullable=True)  # {"skill": weight}
    matching_criteria = Column(JSONB, nullable=True)  # arbitrary JSON rules
    processing_status = Column(String(20), nullable=False, default="PENDING", server_default="PENDING")
    version = Column(Integer, nullable=False, default=1, server_default="1")

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False)
    updated_by = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False)
