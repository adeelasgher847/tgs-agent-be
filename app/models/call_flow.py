from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, Boolean, Index, CheckConstraint, Numeric
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base_class import Base


class CallFlow(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id"), nullable=False)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agent.id"), nullable=False)
    name = Column(String(255), nullable=False)
    direction = Column(String(20), nullable=False)  # inbound | outbound
    welcome_message_type = Column(String(50), nullable=True)
    custom_welcome_message = Column(Text, nullable=True)
    # Circular FK to promptversion — use_alter defers constraint creation
    current_prompt_id = Column(
        UUID(as_uuid=True),
        ForeignKey("promptversion.id", use_alter=True, name="fk_callflow_current_prompt"),
        nullable=True,
    )
    flow_data = Column(JSONB, nullable=True)
    settings = Column(JSONB, nullable=True)
    knowledge_base_ids = Column(JSONB, nullable=True, default=list)

    # A/B prompt testing
    ab_test_enabled = Column(Boolean, default=False, nullable=False, server_default="false")
    ab_prompt_a_id = Column(
        UUID(as_uuid=True),
        ForeignKey("promptversion.id", use_alter=True, name="fk_callflow_ab_prompt_a"),
        nullable=True,
    )
    ab_prompt_b_id = Column(
        UUID(as_uuid=True),
        ForeignKey("promptversion.id", use_alter=True, name="fk_callflow_ab_prompt_b"),
        nullable=True,
    )
    # Fraction of calls routed to variant A (0.10-0.90)
    ab_split_ratio = Column(Numeric(3, 2), default=0.50, nullable=False, server_default="0.50")

    # Cross-session caller memory: inject summaries of a caller's past calls into the prompt
    caller_memory_enabled = Column(Boolean, default=False, nullable=False, server_default="false")
    caller_memory_window = Column(Integer, default=3, nullable=False, server_default="3")

    hipaa_compliance = Column(Boolean, default=False, nullable=False, server_default="false")
    public_access = Column(Boolean, default=False, nullable=False, server_default="false")
    is_deleted = Column(Boolean, default=False, nullable=False, server_default="false")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=True)

    # Relationships
    tenant = relationship("Tenant")
    agent = relationship("Agent")
    prompt_versions = relationship(
        "PromptVersion",
        foreign_keys="[PromptVersion.flow_id]",
        back_populates="call_flow",
        cascade="all, delete-orphan",
        order_by="PromptVersion.created_at.desc()",
    )
    # post_update=True required for circular FK
    current_prompt = relationship(
        "PromptVersion",
        foreign_keys="[CallFlow.current_prompt_id]",
        post_update=True,
    )
    ab_prompt_a = relationship(
        "PromptVersion",
        foreign_keys="[CallFlow.ab_prompt_a_id]",
        post_update=True,
    )
    ab_prompt_b = relationship(
        "PromptVersion",
        foreign_keys="[CallFlow.ab_prompt_b_id]",
        post_update=True,
    )
    call_sessions = relationship("CallSession", back_populates="call_flow")

    __table_args__ = (
        Index("ix_callflow_tenant_id", "tenant_id"),
        Index("ix_callflow_agent_id", "agent_id"),
        CheckConstraint(
            "direction IN ('inbound', 'outbound', 'bidirectional')",
            name="ck_callflow_direction",
        ),
        CheckConstraint(
            "welcome_message_type IS NULL OR "
            "welcome_message_type IN ('user_initiated', 'ai_dynamic', 'ai_custom')",
            name="ck_callflow_welcome_message_type",
        ),
        CheckConstraint(
            "ab_split_ratio > 0 AND ab_split_ratio < 1",
            name="ck_callflow_ab_split_ratio",
        ),
        CheckConstraint(
            "caller_memory_window >= 1 AND caller_memory_window <= 10",
            name="ck_callflow_caller_memory_window",
        ),
    )
