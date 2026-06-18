"""
Append-only audit log for platform-wide configuration changes and data-access events.

Covers both HIPAA § 164.312(b) requirements and enterprise procurement needs.
Rows are never updated or deleted by application code — enforced at the DB level
via a no_update RULE and a no_delete trigger (bypassed by the retention job via
the `app.bypass_audit_delete` session GUC).
"""
from __future__ import annotations

import uuid

from sqlalchemy import JSON, Column, DateTime, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.db.base_class import Base


class AuditLog(Base):
    """Append-only audit trail for all platform mutation events."""

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    tenant_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    actor_api_key_prefix = Column(String(8), nullable=True)
    action = Column(String(128), nullable=False, index=True)
    resource_type = Column(String(64), nullable=True)
    resource_id = Column(UUID(as_uuid=True), nullable=True)
    # Stored as JSONB in PostgreSQL (via migration); JSON type is SQLite-compatible for tests
    old_value = Column(JSON, nullable=True)
    new_value = Column(JSON, nullable=True)
    # Stored as INET in PostgreSQL (via migration); String is SQLite-compatible for tests
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)

    __table_args__ = (
        Index("ix_auditlog_tenant_action", "tenant_id", "action"),
        Index("ix_auditlog_tenant_timestamp", "tenant_id", "timestamp"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<AuditLog action={self.action!r} "
            f"tenant={self.tenant_id} resource={self.resource_type}:{self.resource_id}>"
        )
