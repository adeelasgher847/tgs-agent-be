from sqlalchemy import Column, String, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base_class import Base


class TenantCRMConfig(Base):
    """CRM configuration per tenant - stores API keys and container info for Monday.com, ClickUp, Jira, Trello"""

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id"), nullable=False, index=True)
    
    # CRM type: "monday" | "clickup" | "jira" | "trello"
    crm_type = Column(String(20), nullable=False, index=True)
    
    # Encrypted API key/token (using JWT encryption)
    encrypted_api_key = Column(String(1000), nullable=False)
    
    # Container info (board_id/list_id/project_id)
    container_id = Column(String(200), nullable=True, index=True)
    container_url = Column(String(500), nullable=True)
    
    # Additional config (JSON) for CRM-specific settings
    # For Monday.com: workspace_id
    # For ClickUp: space_id, folder_id
    # For Jira: project_key
    # For Trello: board_id (already in container_id)
    additional_config = Column(String, nullable=True)  # JSON string
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=True)
    
    # Relationships
    tenant = relationship("Tenant", back_populates="crm_configs")
    creator = relationship("User", foreign_keys=[created_by])

    __table_args__ = (
        UniqueConstraint("tenant_id", "crm_type", name="uq_tenant_crm_type"),
    )

    def __repr__(self) -> str:  # pragma: no cover - repr for debugging
        return f"<TenantCRMConfig(tenant_id={self.tenant_id}, crm_type={self.crm_type}, container_id={self.container_id})>"

