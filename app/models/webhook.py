from __future__ import annotations

import uuid

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base_class import Base


class WebhookEndpoint(Base):
    """Tenant-configured outbound webhook endpoint."""

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tenant.id", ondelete="CASCADE"),
        nullable=False,
    )
    url = Column(Text, nullable=False)
    # Signing secret stored encrypted at rest (encrypt_api_key / decrypt_api_key).
    # Not a one-way hash — must be reversible for outbound HMAC signing.
    secret = Column(Text, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True, server_default="true")
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    workspace = relationship("Tenant")
    deliveries = relationship(
        "WebhookDelivery", back_populates="endpoint", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_webhookendpoint_ws", "workspace_id"),
    )


class WebhookDelivery(Base):
    """Log of every outbound webhook delivery attempt."""

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    endpoint_id = Column(
        UUID(as_uuid=True),
        ForeignKey("webhookendpoint.id", ondelete="CASCADE"),
        nullable=False,
    )
    event_type = Column(Text, nullable=False)
    payload = Column(JSONB, nullable=False)
    # "delivered" | "failed" | "retrying"
    status = Column(Text, nullable=False)
    http_status = Column(Integer, nullable=True)
    response_body = Column(Text, nullable=True)
    attempt_count = Column(Integer, nullable=False, default=1, server_default="1")
    last_attempted_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    endpoint = relationship("WebhookEndpoint", back_populates="deliveries")

    __table_args__ = (
        Index("ix_webhookdelivery_ep", "endpoint_id"),
    )
