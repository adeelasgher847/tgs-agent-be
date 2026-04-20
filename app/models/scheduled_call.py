from sqlalchemy import Column, String, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base_class import Base


class ScheduledCall(Base):
    """CRM container configuration per user - stores container info only, not actual call data.
    All tenants of a user share the same container, identified by tenant_id column/field in items.
    Supports Monday.com, ClickUp, Jira, and Trello."""

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False, index=True)
    
    # Reference to global CRM configuration (column name kept for backward compatibility)
    tenant_crm_config_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # Generic container info (works for all CRMs)
    # Note: nullable=True in DB for backward compatibility, but should always have values for new records
    crm_container_id = Column(String(200), nullable=True, index=True)  # board_id/list_id/project_id
    crm_container_url = Column(String(500), nullable=True)
    crm_type = Column(String(20), nullable=True, index=True)  # "monday" | "clickup" | "jira" | "trello"
    
    # Legacy fields (for backward compatibility, can be removed later)
    monday_board_id = Column(String(50), nullable=True, index=True)
    monday_board_url = Column(String(500), nullable=True)
    # Optional relation to resume interview scheduling entries.
    # NULL means this row is a CRM container mapping row.
    resume_interview_id = Column(
        UUID(as_uuid=True),
        ForeignKey("resumeinterview.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    user = relationship("User", back_populates="scheduled_calls")
    resume_interview = relationship("ResumeInterview", back_populates="scheduled_calls")
    # Note: tenant_crm_config relationship removed to avoid FK validation issues
    # Access CRMConfig via: crm_config_service.get_crm_config_by_id(db, scheduled_call.tenant_crm_config_id)

    # One board per user per CRM config. Partial uniques (for NULL handling) are in migration.
    __table_args__ = ()

    def __repr__(self) -> str:  # pragma: no cover - repr for debugging
        return f"<ScheduledCall(user_id={self.user_id}, crm_type={self.crm_type}, container_id={self.crm_container_id})>"

