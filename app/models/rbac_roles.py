from sqlalchemy import Column, String, DateTime, Index, text, ForeignKey, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base_class import Base

class RbacRole(Base):
    __tablename__ = "rbac_roles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    role = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True) # Soft-delete convention

    # Relationship
    tenant = relationship("Tenant", back_populates="rbac_roles")

    __table_args__ = (
        Index("idx_rbac_roles_workspace_user", "workspace_id", "user_id"),
        Index("uq_rbac_roles_workspace_user_active", "workspace_id", "user_id", unique=True, postgresql_where=text("deleted_at IS NULL")),
        CheckConstraint(
            "role IN ('admin', 'manager', 'config_only', 'read_only', 'billing_only')", 
            name="chk_rbac_roles_role"
        ),
    )