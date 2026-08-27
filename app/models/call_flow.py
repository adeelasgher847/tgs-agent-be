from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    Integer,
    ForeignKey,
    Boolean,
    Index,
    CheckConstraint,
    Numeric,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func, text as sa_text
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
        ForeignKey(
            "promptversion.id", use_alter=True, name="fk_callflow_current_prompt"
        ),
        nullable=True,
    )
    flow_data = Column(JSONB, nullable=True)
    # Pre-compiled decision tree derived from flow_data — {node_id: {type, data, next_nodes}}
    compiled_plan = Column(JSONB, nullable=True)
    settings = Column(JSONB, nullable=True)
    knowledge_base_ids = Column(JSONB, nullable=True, default=list)

    # A/B prompt testing
    ab_test_enabled = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )
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
    ab_split_ratio = Column(
        Numeric(3, 2), default=0.50, nullable=False, server_default="0.50"
    )

    # Cross-session caller memory: inject summaries of a caller's past calls into the prompt
    caller_memory_enabled = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )
    caller_memory_window = Column(
        Integer, default=3, nullable=False, server_default="3"
    )

    # Post Call Actions: email a call summary and/or notify the tenant's business
    # owner (is_creator user) after a call completes
    email_summary_enabled = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )
    email_summary_recipients = Column(JSONB, nullable=True, default=list)
    summary_to_business_owner_enabled = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )

    # Post Call Actions: post a call summary to a Slack channel after a call
    # completes. Per-call-flow opt-in — a workspace connecting Slack (see
    # WorkspaceIntegration, provider="slack") does not activate this for every
    # call flow. If slack_channel_id is null, the send-time fallback is the
    # workspace's default_channel_id in WorkspaceIntegration.extra_metadata.
    # Has no effect on inbound CRM-sync calls — mirrors email_summary_enabled's
    # behavior for that call type (see call_session_service.py's post-call hooks).
    slack_summary_enabled = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )
    slack_channel_id = Column(String(50), nullable=True)
    # Denormalized display name of slack_channel_id, so the frontend doesn't
    # need an extra Slack API round-trip to show the selected channel in the UI.
    slack_channel_name = Column(String(255), nullable=True)

    # Post-Call Analysis: tenant-defined variables extracted from each completed
    # call via a dedicated LLM pass, independent of the fixed
    # summary/sentiment/recommendations fields computed by
    # voice_analysis_service.analyze_call_transcript. Empty list = feature off;
    # the automatic call-summary generation is unaffected either way.
    post_call_analysis_variables = Column(
        # No explicit `::jsonb` cast here (unlike the Alembic migration for
        # this column) — Postgres infers the cast from the column's own type
        # in a DEFAULT clause, and omitting it keeps this portable to the
        # SQLite dialect used by the unit-test suite (SQLite's parser does
        # not understand `::` cast syntax at all).
        JSONB,
        nullable=False,
        default=list,
        server_default=sa_text("'[]'"),
    )
    post_call_analysis_model = Column(String(100), nullable=True)

    # System Webhooks: Pre-Inbound Call Webhook — fired before an inbound call
    # connects; response `{"variables": {...}}` gets injected into the agent's
    # prompt/greeting via `{{key}}` placeholders. Fail-open on any failure.
    pre_inbound_webhook_url = Column(String(2048), nullable=True)
    # pgcrypto ciphertext (encrypt_webhook_headers) of the full headers dict JSON
    pre_inbound_webhook_headers_encrypted = Column(Text, nullable=True)
    # [{key, value}] — values may contain `{{...}}` template tokens, not secret
    pre_inbound_webhook_query_params = Column(
        JSONB,
        nullable=False,
        default=list,
        server_default=sa_text("'[]'"),
    )
    # Flat tenant-defined key/value map, available at render time under both
    # `_metadata.*` and `_variable.*` template namespaces
    pre_inbound_webhook_static_metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=sa_text("'{}'"),
    )

    # System Webhooks: Dynamic Inbound Call Routing — depends on the
    # Pre-Inbound webhook above; if its response includes a valid
    # `variables.agent_id`, route the call to that agent instead of the
    # number's default agent.
    dynamic_inbound_routing_enabled = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )

    # System Webhooks: Post-Call Webhook — fired after a call ends with
    # `{callId, agentId, timestamp, data}`; optional custom payload mode lets
    # the tenant define their own JSON body using `{{field}}` tokens.
    post_call_webhook_url = Column(String(2048), nullable=True)
    post_call_webhook_headers_encrypted = Column(Text, nullable=True)
    post_call_webhook_query_params = Column(
        JSONB,
        nullable=False,
        default=list,
        server_default=sa_text("'[]'"),
    )
    post_call_webhook_custom_payload_enabled = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )
    # Arbitrary JSON object with `{{field}}` string tokens as leaf values
    post_call_webhook_custom_payload_template = Column(JSONB, nullable=True)

    # System Webhooks: Status Webhook — fired on call lifecycle sub-events
    # (connect, transfer, end) with a status/outcome payload.
    status_webhook_enabled = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )
    status_webhook_url = Column(String(2048), nullable=True)
    status_webhook_headers_encrypted = Column(Text, nullable=True)
    status_webhook_query_params = Column(
        JSONB,
        nullable=False,
        default=list,
        server_default=sa_text("'[]'"),
    )

    # Voicemail Detection Settings: configure behavior when a voicemail system is detected
    voicemail_detection_enabled = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )
    voicemail_action = Column(
        String(50), default="hang_up", nullable=False, server_default="hang_up"
    )
    voicemail_message = Column(Text, nullable=True)
    voicemail_advanced_detection_enabled = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )
    voicemail_detection_timeout = Column(
        Integer, default=5, nullable=False, server_default="5"
    )

    # Call Screening Detection Settings: action when automated screener (Google, Samsung, IVR) answers
    call_screening_action = Column(
        String(50), default="respond", nullable=False, server_default="respond"
    )

    # Disable Metadata: whether to strip metadata from outbound API and webhook payloads
    disable_metadata = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )

    # IVR Phone Tree Navigation Settings
    ivr_enabled = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )
    ivr_action = Column(
        String(50), default="dial_through", nullable=False, server_default="dial_through"
    )
    ivr_navigation_mode = Column(
        String(50), default="let_ai_converse", nullable=False, server_default="let_ai_converse"
    )
    ivr_max_attempts = Column(
        Integer, default=3, nullable=False, server_default="3"
    )
    ivr_keypress_delay = Column(
        Integer, default=8, nullable=False, server_default="8"
    )
    ivr_priority_list = Column(
        JSONB, default=list, nullable=False, server_default=sa_text("'[]'")
    )
    ivr_wait_on_hold = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )
    ivr_max_hold_time = Column(
        Integer, default=120, nullable=False, server_default="120"
    )

    # In-Call DTMF Keypad Detection Settings
    dtmf_enabled = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )
    dtmf_button_press_delay = Column(
        Integer, default=2, nullable=False, server_default="2"
    )
    dtmf_allow_caller_interruption = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )
    dtmf_max_digits = Column(
        Integer, default=50, nullable=False, server_default="50"
    )
    dtmf_allowed_exceeded_attempts = Column(
        Integer, default=10, nullable=False, server_default="10"
    )
    dtmf_exceeded_action = Column(
        String(50), default="end_call", nullable=False, server_default="end_call"
    )
    dtmf_end_call_message = Column(
        Text,
        nullable=True,
        default="You've reached the maximum number of inputs allowed for this call.",
        server_default="You've reached the maximum number of inputs allowed for this call.",
    )

    # ── Call Timing & Silence Detection Settings ──
    silence_timeout = Column(
        Integer, default=10, nullable=False, server_default="10"
    )
    end_call_after_reminder = Column(
        Integer, default=10, nullable=False, server_default="10"
    )
    reminder_retries = Column(
        Integer, default=1, nullable=False, server_default="1"
    )
    reminder_messages = Column(
        JSONB, default=list, nullable=False, server_default=sa_text("'[]'")
    )
    max_call_duration = Column(
        Integer, default=1800, nullable=False, server_default="1800"
    )
    max_duration_message = Column(
        Text,
        nullable=True,
        default="I appreciate the conversation, but we've reached our time limit for this call.",
        server_default="I appreciate the conversation, but we've reached our time limit for this call.",
    )

    # ── Inbound Call Redirection & Forwarding Settings ──
    redirect_inbound_calls_enabled = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )
    redirect_forward_phone_number = Column(String(50), nullable=True)
    redirect_conditions = Column(
        JSONB, default=list, nullable=False, server_default=sa_text("'[]'")
    )
    redirect_speak_message_enabled = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )
    redirect_message = Column(Text, nullable=True)

    # ── Inbound Rules & Blocklist Rule Set ──
    inbound_rule_set_id = Column(
        UUID(as_uuid=True),
        ForeignKey("inboundruleset.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # ── Call Recording Settings ──
    recording_enabled = Column(
        Boolean, default=True, nullable=False, server_default="true"
    )
    public_recording_enabled = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )
    faster_inbound_pickup = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )
    stop_recording_on_transfer = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )

    # ── Compliance & Detection Settings ──
    compliance_monitoring_enabled = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )
    anti_bot_detection_enabled = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )
    terminate_on_fake_voice = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )

    # ── Data Retention Policy Settings ──
    retention_policy_enabled = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )
    retention_transcript_enabled = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )
    retention_transcript_days = Column(
        Integer, default=30, nullable=False, server_default="30"
    )
    retention_summary_enabled = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )
    retention_summary_days = Column(
        Integer, default=30, nullable=False, server_default="30"
    )
    retention_recording_enabled = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )
    retention_recording_days = Column(
        Integer, default=30, nullable=False, server_default="30"
    )

    hipaa_compliance = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )
    public_access = Column(
        Boolean, default=False, nullable=False, server_default="false"
    )
    # "active" flows can be used to initiate outbound calls; "inactive" ones are rejected
    # at dispatch time (see voice_call_service.initiate_call) without being deleted.
    status = Column(
        String(20), default="active", nullable=False, server_default="active"
    )
    is_deleted = Column(Boolean, default=False, nullable=False, server_default="false")
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=True,
    )

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
    inbound_rule_set = relationship(
        "InboundRuleSet",
        foreign_keys=[inbound_rule_set_id],
        back_populates="call_flows",
    )

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
        CheckConstraint(
            "status IN ('active', 'inactive')",
            name="ck_callflow_status",
        ),
    )
