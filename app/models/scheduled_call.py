from sqlalchemy import Column, String, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base_class import Base


class ScheduledCall(Base):
    """Monday.com board configuration per user - stores board info only, not actual call data.
    All tenants of a user share the same board, identified by tenant_id column in items."""

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False, unique=True, index=True)
    monday_board_id = Column(String(50), nullable=False, index=True)
    monday_board_url = Column(String(500), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    user = relationship("User", back_populates="scheduled_call")

    __table_args__ = (UniqueConstraint("user_id", name="uq_scheduledcall_user_id"),)

    def __repr__(self) -> str:  # pragma: no cover - repr for debugging
        return f"<ScheduledCall(user_id={self.user_id}, board_id={self.monday_board_id})>"

