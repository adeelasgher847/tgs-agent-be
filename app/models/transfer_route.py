from sqlalchemy import Column, String, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base_class import Base


class TransferRoute(Base):
    """
    Tenant-scoped human transfer destination (cold dial or warm conference).
    Table name follows Base: TransferRoute -> transferroute.
    """

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id"), nullable=False, index=True)
    friendly_name = Column(String(255), nullable=False)
    phone_number = Column(String(20), nullable=False)
    transfer_type = Column(String(16), nullable=False, server_default="cold")
    is_deleted = Column(Boolean, nullable=False, server_default="false", default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    tenant = relationship("Tenant", back_populates="transfer_routes")
    agents = relationship("Agent", back_populates="transfer_route")
