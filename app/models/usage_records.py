from sqlalchemy import Column, DateTime, Numeric, Index, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base_class import Base


class UsageRecords(Base):
    __tablename__ = "usage_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id", ondelete="CASCADE"), nullable=False)
    call_id = Column(UUID(as_uuid=True), ForeignKey("callsessions.id", ondelete="SET NULL"), nullable=True) 
    billable_minutes = Column(Numeric(10, 2), nullable=False)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True) # Soft-delete convention

    # Relationship
    tenant = relationship("Tenant", back_populates="usage_records")

    __table_args__ = (
        Index("idx_usage_records_workspace_recorded_at", "workspace_id", "recorded_at"),
    )