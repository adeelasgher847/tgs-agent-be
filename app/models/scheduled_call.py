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
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False, unique=True, index=True)
    
    # Reference to tenant's CRM configuration
    # Note: ForeignKey removed - relationship uses explicit primaryjoin instead
    tenant_crm_config_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # Generic container info (works for all CRMs)
    crm_container_id = Column(String(200), nullable=False, index=True)  # board_id/list_id/project_id
    crm_container_url = Column(String(500), nullable=False)
    crm_type = Column(String(20), nullable=False, index=True)  # "monday" | "clickup" | "jira" | "trello"
    
    # Legacy fields (for backward compatibility, can be removed later)
    monday_board_id = Column(String(50), nullable=True, index=True)
    monday_board_url = Column(String(500), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    user = relationship("User", back_populates="scheduled_call")
    tenant_crm_config = relationship(
        "TenantCRMConfig",
        primaryjoin="ScheduledCall.tenant_crm_config_id == TenantCRMConfig.id"
    )

    __table_args__ = (UniqueConstraint("user_id", name="uq_scheduledcall_user_id"),)

    def __repr__(self) -> str:  # pragma: no cover - repr for debugging
        return f"<ScheduledCall(user_id={self.user_id}, crm_type={self.crm_type}, container_id={self.crm_container_id})>"

