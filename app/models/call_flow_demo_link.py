from __future__ import annotations

import uuid

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, Numeric, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base_class import Base


class CallFlowDemoLink(Base):
    """Named, shareable demo link for a call flow.

    Origin-unrestricted (unlike the embedded-widget public-call-token flow,
    which is gated by AllowedDomain) — access is controlled purely by
    possession of the opaque `token`, expiry, and usage-budget fields below.
    """

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Not indexed here via column(index=True) — the explicit Index() entries in
    # __table_args__ below cover these columns. Declaring both produces two
    # identically-named indexes, which SQLite (and Postgres) reject as a
    # duplicate object.
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id"), nullable=False)
    call_flow_id = Column(UUID(as_uuid=True), ForeignKey("callflow.id"), nullable=False)

    name = Column(String(255), nullable=True)
    # Opaque unguessable public identifier for the link, generated via
    # secrets.token_urlsafe(32) at creation time in the service layer.
    token = Column(String, nullable=False, unique=True, index=True)

    is_active = Column(Boolean, nullable=False, default=True, server_default="true")
    expires_at = Column(DateTime(timezone=True), nullable=True)

    per_user_limit_minutes = Column(Numeric, nullable=True)
    total_budget_minutes = Column(Numeric, nullable=True)
    # Incremented by CallSessionService._record_demo_link_usage(), called
    # from update_call_session_status() when app.voice.livekit_browser_call_handler
    # .run_livekit_browser_call() finalizes a demo-link CallSession as the
    # LiveKit room disconnects — this does decrement from real call activity.
    total_minutes_used = Column(Numeric, nullable=False, default=0, server_default="0")

    created_by_user_id = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=True)

    # Relationships
    tenant = relationship("Tenant")
    call_flow = relationship("CallFlow")
    created_by_user = relationship("User")
    visitor_usages = relationship(
        "CallFlowDemoLinkVisitorUsage",
        back_populates="demo_link",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_callflowdemolink_tenant_id", "tenant_id"),
        Index("ix_callflowdemolink_call_flow_id", "call_flow_id"),
    )
