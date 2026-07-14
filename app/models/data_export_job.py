from __future__ import annotations

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base_class import Base


class DataExportJob(Base):
    """Tracks one GDPR data-portability export (ARQ job -> ZIP in S3)."""

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id", ondelete="CASCADE"), nullable=False)
    requested_by_user_id = Column(UUID(as_uuid=True), ForeignKey("user.id", ondelete="SET NULL"), nullable=True)

    # processing -> ready | error
    status = Column(String(20), nullable=False, default="processing", server_default="processing")

    s3_path = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    workspace = relationship("Tenant")

    __table_args__ = (
        Index("ix_dataexportjob_workspace_id", "workspace_id"),
        Index("ix_dataexportjob_status", "status"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<DataExportJob id={self.id} workspace={self.workspace_id} status={self.status}>"
