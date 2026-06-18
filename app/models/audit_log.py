"""
Immutable audit log for HIPAA compliance events.

Every security-relevant action (HIPAA toggle, KMS key update, BAA change,
recording access, etc.) is appended here.  Rows are never updated or deleted
— this table is append-only by convention and enforced via application logic.

HIPAA § 164.312(b): Audit controls — record and examine activity in systems
that contain or use electronic protected health information (ePHI).
"""
from __future__ import annotations

import uuid

from sqlalchemy import Column, DateTime, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.db.base_class import Base


class AuditLog(Base):
    """Append-only audit trail for HIPAA and security events."""

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    tenant_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    action = Column(String(128), nullable=False, index=True)
    resource_type = Column(String(64), nullable=True)
    resource_id = Column(UUID(as_uuid=True), nullable=True)
    old_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)  # max IPv6 = 45 chars
    user_agent = Column(String(512), nullable=True)

    __table_args__ = (
        Index("ix_auditlog_tenant_action", "tenant_id", "action"),
        Index("ix_auditlog_tenant_timestamp", "tenant_id", "timestamp"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<AuditLog action={self.action!r} "
            f"tenant={self.tenant_id} resource={self.resource_type}:{self.resource_id}>"
        )