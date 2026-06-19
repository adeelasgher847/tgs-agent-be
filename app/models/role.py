from sqlalchemy import Column, String, DateTime, Index, text, ForeignKey, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base_class import Base


class Role(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    role = Column(String, nullable=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    tenant = relationship("Tenant", back_populates="rbac_roles")

    __table_args__ = (
        CheckConstraint(
            "role IN ('admin', 'manager', 'config_only', 'read_only', 'billing_only')",
            name="chk_role_role",
        ),
    )