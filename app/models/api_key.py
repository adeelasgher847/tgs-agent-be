from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base_class import Base


class Apikey(Base):
    """Tenant-scoped API key for machine-to-machine auth (stored as SHA-256 hash)."""

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    key_prefix = Column(String(32), nullable=False)  # masked display, e.g. sk_ab••••xy12
    key_hash = Column(String(64), nullable=False, unique=True)  # SHA-256 hex digest
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_used_at = Column(DateTime(timezone=True), nullable=True)

    tenant = relationship("Tenant", back_populates="api_keys")

    __table_args__ = (
        Index("ix_apikey_key_hash_tenant_id", "key_hash", "tenant_id"),
        Index("ix_apikey_tenant_id", "tenant_id"),
    )
