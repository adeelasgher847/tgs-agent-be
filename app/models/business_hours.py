from sqlalchemy import Column, String, DateTime, Integer, Boolean, Time, UniqueConstraint, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base_class import Base


class BusinessHours(Base):
    """Working hours per weekday per tenant. One row per (tenant, day)."""

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id"), nullable=False, index=True)
    day_of_week = Column(Integer, nullable=False)       # 0=Monday … 6=Sunday
    open_time = Column(Time, nullable=True)             # None when is_closed=True
    close_time = Column(Time, nullable=True)
    is_closed = Column(Boolean, nullable=False, server_default="false")
    timezone = Column(String(60), nullable=False, server_default="UTC")
    slot_duration_minutes = Column(Integer, nullable=False, server_default="30")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    tenant = relationship("Tenant", back_populates="business_hours")

    __table_args__ = (
        UniqueConstraint("tenant_id", "day_of_week", name="uq_businesshours_tenant_day"),
    )
