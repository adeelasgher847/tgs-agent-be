from __future__ import annotations

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Index, Numeric, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base_class import Base


class CallFlowDemoLinkVisitorUsage(Base):
    """Per-anonymous-visitor minute consumption against a demo link's
    per_user_limit_minutes budget.

    `visitor_id` is a client-supplied opaque identifier, not a FK — there is
    no `user` row for anonymous public demo-link visitors.
    """

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # demo_link_id is not indexed here via column(index=True) — see the
    # explicit Index() in __table_args__ below; declaring both would create
    # two identically-named indexes (rejected as duplicates by SQLite/Postgres).
    demo_link_id = Column(
        UUID(as_uuid=True), ForeignKey("callflowdemolink.id"), nullable=False
    )
    visitor_id = Column(String, nullable=False, index=True)

    # TODO(known limitation, not missing logic): see the matching comment
    # on CallFlowDemoLink.total_minutes_used — same accepted gap, same
    # accounting hook, currently untriggered in production.
    minutes_used = Column(Numeric, nullable=False, default=0, server_default="0")

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=True)

    demo_link = relationship("CallFlowDemoLink", back_populates="visitor_usages")

    __table_args__ = (
        Index("ix_callflowdemolinkvisitorusage_demo_link_id", "demo_link_id"),
        UniqueConstraint("demo_link_id", "visitor_id", name="uq_calldemolink_visitor"),
    )
