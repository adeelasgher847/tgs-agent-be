"""Inbound Rules & Number Blocking Rule Sets models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class InboundRuleSet(Base):
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
    tenant_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tenant.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    is_deleted = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )
    created_by = Column(
        UUID(as_uuid=True), ForeignKey("user.id"), nullable=True
    )
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    rules = relationship(
        "InboundRule",
        back_populates="rule_set",
        cascade="all, delete-orphan",
        order_by="InboundRule.created_at.desc()",
    )
    call_flows = relationship("CallFlow", back_populates="inbound_rule_set")


class InboundRule(Base):
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
    tenant_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tenant.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    rule_set_id = Column(
        UUID(as_uuid=True),
        ForeignKey("inboundruleset.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    phone_number_pattern = Column(String(50), nullable=False)
    normalized_digits = Column(String(50), nullable=False, index=True)
    label = Column(String(100), nullable=True)
    action = Column(
        String(20), default="deny", nullable=False, server_default="deny"
    )
    is_deleted = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    rule_set = relationship("InboundRuleSet", back_populates="rules")

    __table_args__ = (
        Index(
            "ix_inboundrule_tenant_normalized",
            "tenant_id",
            "normalized_digits",
        ),
        Index(
            "ix_inboundrule_set_normalized",
            "rule_set_id",
            "normalized_digits",
        ),
    )
