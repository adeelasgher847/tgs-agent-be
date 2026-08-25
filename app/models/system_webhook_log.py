from __future__ import annotations

import uuid

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base_class import Base


class SystemWebhookDeliveryLog(Base):
    """Delivery outcome log for Call Flow System Webhooks (pre_inbound / post_call /
    status). Deliberately does NOT store request headers/params/body — those may
    carry tenant secrets — only outcome metadata for the "Test Webhook" button
    result and tenant-facing debugging.
    """

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tenant.id"),
        nullable=False,
    )
    call_flow_id = Column(
        UUID(as_uuid=True),
        ForeignKey("callflow.id"),
        nullable=False,
    )
    # Pre-inbound webhook can fire before/without a durable session in some
    # failure paths; post_call/status always have one.
    call_session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("callsession.id"),
        nullable=True,
    )
    webhook_kind = Column(String(20), nullable=False)
    # e.g. connect/transfer/end for status webhooks; null for pre_inbound/post_call
    event_type = Column(String(50), nullable=True)
    url = Column(String(2048), nullable=False)
    status = Column(String(20), nullable=False)
    status_code = Column(Integer, nullable=True)
    # Caller must truncate before insert — no DB-side truncation
    response_body = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    attempt_count = Column(Integer, nullable=False, default=1, server_default="1")
    duration_ms = Column(Integer, nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    tenant = relationship("Tenant")
    call_flow = relationship("CallFlow")
    call_session = relationship("CallSession")

    __table_args__ = (
        Index("ix_systemwebhookdeliverylog_tenant_id", "tenant_id"),
        Index(
            "ix_systemwebhookdeliverylog_call_flow_id_created_at",
            "call_flow_id",
            "created_at",
        ),
        CheckConstraint(
            "webhook_kind IN ('pre_inbound', 'post_call', 'status')",
            name="ck_systemwebhookdeliverylog_webhook_kind",
        ),
        CheckConstraint(
            "status IN ('success', 'failed', 'timeout')",
            name="ck_systemwebhookdeliverylog_status",
        ),
    )
