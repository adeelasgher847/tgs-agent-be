from sqlalchemy import Column, String, DateTime, Numeric, Index, text, ForeignKey, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base_class import Base

class BrandingConfig(Base):
    __tablename__ = "branding_configs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id", ondelete="CASCADE"), unique=True, nullable=False)
    logo_url = Column(String, nullable=True)
    primary_colour = Column(String(7), nullable=True)  # CHAR(7) requirement
    display_name = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True) # Soft-delete convention

    # Relationship
    tenant = relationship("Tenant", back_populates="branding_config")