from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    field_validator,
    model_validator,
)

from app.schemas.prompt_version import PromptVersionOut
from app.utils.ssrf import assert_public_url


class DirectionEnum(str, Enum):
    inbound = "inbound"
    outbound = "outbound"
    bidirectional = "bidirectional"


class WelcomeMessageTypeEnum(str, Enum):
    user_initiated = "user_initiated"
    ai_dynamic = "ai_dynamic"
    ai_custom = "ai_custom"


class CallFlowStatusEnum(str, Enum):
    active = "active"
    inactive = "inactive"


class FlowDataSchema(BaseModel):
    """Structural validation for flowData JSONB — future visual-editor format."""

    model_config = ConfigDict(extra="allow")

    nodes: List[Any] = Field(default_factory=list)
    edges: List[Any] = Field(default_factory=list)


class AgentRef(BaseModel):
    """Embedded agent snapshot in flow responses."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str


class CallFlowCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(..., min_length=1, max_length=255)
    direction: DirectionEnum
    agent_id: uuid.UUID = Field(..., alias="agentId")
    welcome_message_type: WelcomeMessageTypeEnum | None = Field(
        None, alias="welcomeMessageType"
    )
    custom_welcome_message: str | None = Field(None, alias="customWelcomeMessage")
    prompt: str | None = None
    notes: str | None = None  # notes for the initial prompt version
    flow_data: FlowDataSchema | None = Field(None, alias="flowData")
    # Free-form flow settings. Recognized keys include:
    #   calendly_integration_enabled: bool — routes booking through Gemini
    #   function-calling + Calendly instead of the legacy [BOOK_APPOINTMENT:...]
    #   regex-token pipeline (see app/voice/booking_mixin.py::_calendly_enabled).
    settings: Dict[str, Any] | None = None
    status: CallFlowStatusEnum = CallFlowStatusEnum.active


class CallFlowUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str | None = Field(None, min_length=1, max_length=255)
    direction: DirectionEnum | None = None
    agent_id: uuid.UUID | None = Field(None, alias="agentId")
    welcome_message_type: WelcomeMessageTypeEnum | None = Field(
        None, alias="welcomeMessageType"
    )
    custom_welcome_message: str | None = Field(None, alias="customWelcomeMessage")
    prompt: str | None = None
    notes: str | None = None  # notes for the new prompt version
    current_prompt_id: uuid.UUID | None = Field(None, alias="currentPromptId")
    flow_data: FlowDataSchema | None = Field(None, alias="flowData")
    settings: Dict[str, Any] | None = None
    status: CallFlowStatusEnum | None = None

    @model_validator(mode="after")
    def prompt_and_rollback_exclusive(self) -> "CallFlowUpdate":
        if self.prompt and self.current_prompt_id:
            raise ValueError(
                "'prompt' and 'currentPromptId' are mutually exclusive — "
                "use 'prompt' to create a new version or 'currentPromptId' to roll back, not both."
            )
        return self


class CallFlowSettingsUpdate(BaseModel):
    """Request body for ``PUT /api/v1/call-flows/{id}/settings``."""

    model_config = ConfigDict(extra="forbid")

    public_access: bool = Field(..., alias="public_access")


class CallerMemorySettingsUpdate(BaseModel):
    """Request body for ``PUT /api/v2/flows/{flow_id}/caller-memory-settings``."""

    model_config = ConfigDict(extra="forbid")

    caller_memory_enabled: bool
    caller_memory_window: int = Field(..., ge=1, le=10)


class CallerMemorySettingsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    caller_memory_enabled: bool
    caller_memory_window: int


class PostCallActionsSettingsUpdate(BaseModel):
    """Request body for ``PUT /api/v2/flows/{flow_id}/post-call-actions-settings``."""

    model_config = ConfigDict(extra="forbid")

    email_summary_enabled: bool = Field(
        ...,
        description=(
            "When true, an email with the call summary and extracted variables is sent "
            "to `email_summary_recipients` after each completed call on this flow."
        ),
    )
    email_summary_recipients: List[EmailStr] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "Recipient email addresses for the post-call summary email. Ignored when "
            "`email_summary_enabled` is false. Max 10 addresses. Each entry must be a "
            "valid email address; invalid entries are rejected with a 422."
        ),
    )
    summary_to_business_owner_enabled: bool = Field(
        ...,
        description=(
            "When true, the same post-call summary email is also sent to the tenant's "
            "business owner (the workspace creator's account email) — no separate email "
            "field is needed for this recipient."
        ),
    )
    slack_summary_enabled: bool = Field(
        default=False,
        description=(
            "When true, a call summary is posted to a Slack channel after each completed "
            "call on this flow. Requires the workspace to have connected Slack (see "
            "`/api/v1/integrations/slack`). Falls back to the workspace's default channel "
            "if `slack_channel_id` is not set. Has no effect on inbound CRM-sync calls "
            "(mirrors `email_summary_enabled`'s behavior for that call type)."
        ),
    )
    slack_channel_id: str | None = Field(
        default=None,
        max_length=50,
        description=(
            "Slack channel ID to post the summary to for this flow, overriding the "
            "workspace default. Must be set together with `slack_channel_name`, or both "
            "left null to fall back to the workspace default channel."
        ),
    )
    slack_channel_name: str | None = Field(
        default=None,
        max_length=255,
        description="Denormalized display name of `slack_channel_id`, for the dashboard UI.",
    )

    @model_validator(mode="after")
    def validate_slack_channel_pair(self) -> "PostCallActionsSettingsUpdate":
        if bool(self.slack_channel_id) != bool(self.slack_channel_name):
            raise ValueError(
                "slack_channel_id and slack_channel_name must both be set or both be null"
            )
        return self


class PostCallActionsSettingsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    email_summary_enabled: bool = False
    email_summary_recipients: List[str] = Field(default_factory=list)
    summary_to_business_owner_enabled: bool = False
    slack_summary_enabled: bool = False
    slack_channel_id: str | None = None
    slack_channel_name: str | None = None


class PostCallAnalysisVariableSpec(BaseModel):
    """A single tenant-defined variable to extract from a completed call."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description="Identifier used as the extraction result's JSON key, e.g. service_type.",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "What the model should extract, e.g. 'Extract whether the caller is seeking "
            "residential plumbing, commercial plumbing, or industrial supplies.'"
        ),
    )


class PostCallAnalysisSettingsUpdate(BaseModel):
    """Request body for ``PUT /api/v2/flows/{flow_id}/post-call-analysis-settings``."""

    model_config = ConfigDict(extra="forbid")

    variables_to_extract: List[PostCallAnalysisVariableSpec] = Field(
        default_factory=list,
        max_length=25,
        description=(
            "Custom variables to extract from each completed call. Empty disables this "
            "feature; the automatic call summary keeps running unchanged."
        ),
    )
    analysis_model: str | None = Field(
        None,
        min_length=1,
        max_length=100,
        description=(
            "Model name for extraction; must be an active model-catalog entry. Falls back "
            "to the flow's agent model, then a built-in chain, when unset."
        ),
    )

    @field_validator("analysis_model")
    @classmethod
    def _strip_analysis_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("analysisModel must not be blank")
        return cleaned


class PostCallAnalysisSettingsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    variables_to_extract: List[PostCallAnalysisVariableSpec] = Field(default_factory=list)
    analysis_model: str | None = None


class VoicemailActionEnum(str, Enum):
    HANG_UP = "hang_up"
    LEAVE_MESSAGE = "leave_message"
    CONTINUE = "continue"


class VoicemailSettingsUpdate(BaseModel):
    """Request body for ``PUT /api/v2/flows/{flow_id}/voicemail-settings``."""

    model_config = ConfigDict(extra="forbid")

    voicemail_detection_enabled: bool = Field(
        ...,
        description="Whether voicemail detection is enabled for this call flow.",
    )
    voicemail_action: VoicemailActionEnum = Field(
        default=VoicemailActionEnum.HANG_UP,
        description="Action to take when a voicemail system is detected: hang_up, leave_message, or continue.",
    )
    voicemail_message: str | None = Field(
        default=None,
        max_length=500,
        description="Optional message to leave if voicemail_action is 'leave_message'.",
    )
    voicemail_advanced_detection_enabled: bool = Field(
        default=False,
        description="Whether advanced detection (Twilio Answering Machine Detection - AMD) is enabled.",
    )
    voicemail_detection_timeout: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Voicemail detection timeout in seconds (1–30).",
    )

    @field_validator("voicemail_message")
    @classmethod
    def _strip_voicemail_message(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class VoicemailSettingsResponse(BaseModel):
    """Response body for ``GET/PUT /api/v2/flows/{flow_id}/voicemail-settings``."""

    model_config = ConfigDict(from_attributes=True)

    voicemail_detection_enabled: bool = False
    voicemail_action: str = "hang_up"
    voicemail_message: str | None = None
    voicemail_advanced_detection_enabled: bool = False
    voicemail_detection_timeout: int = 5


class CallScreeningActionEnum(str, Enum):
    RESPOND = "respond"
    HANG_UP = "hang_up"


class CallScreeningSettingsUpdate(BaseModel):
    """Request body for ``PUT /api/v2/flows/{flow_id}/call-screening-settings``."""

    model_config = ConfigDict(extra="forbid")

    call_screening_action: CallScreeningActionEnum = Field(
        default=CallScreeningActionEnum.RESPOND,
        description="Action to take when an automated call screener is detected: respond or hang_up.",
    )


class CallScreeningSettingsResponse(BaseModel):
    """Response body for ``GET/PUT /api/v2/flows/{flow_id}/call-screening-settings``."""

    model_config = ConfigDict(from_attributes=True)

    call_screening_action: str = "respond"


class MetadataSettingsUpdate(BaseModel):
    """Request body for ``PUT /api/v2/flows/{flow_id}/metadata-settings``."""

    model_config = ConfigDict(extra="forbid")

    disable_metadata: bool = Field(
        ...,
        description="Whether to strip metadata from outbound API and webhook payloads.",
    )


class MetadataSettingsResponse(BaseModel):
    """Response body for ``GET/PUT /api/v2/flows/{flow_id}/metadata-settings``."""

    model_config = ConfigDict(from_attributes=True)

    disable_metadata: bool = False


class SystemWebhooksSettingsUpdate(BaseModel):
    """Request body for ``PUT /api/v2/flows/{flow_id}/system-webhooks-settings``.

    Covers all four System Webhooks sub-features in one PUT: Pre-Inbound Call
    Webhook, Dynamic Inbound Call Routing, Post-Call Webhook, Status Webhook.

    Full-replace semantics for every field EXCEPT the three `*_headers`
    fields: those are plaintext-in / never-plaintext-out (the response only
    reports whether headers are configured, not their values), so `None`
    means "leave the stored headers unchanged" rather than "clear them" — see
    `CallFlowService.update_system_webhooks_settings` for the exact rule
    (an explicit `{}` DOES clear them). Every other field is a true
    full-replace: send the current value for anything you don't want to change.
    """

    model_config = ConfigDict(extra="forbid")

    # Pre-Inbound Call Webhook
    pre_inbound_webhook_url: str | None = None
    pre_inbound_webhook_headers: Dict[str, str] | None = None
    pre_inbound_webhook_query_params: List[Dict[str, str]] | None = None
    pre_inbound_webhook_static_metadata: Dict[str, str] | None = None

    # Dynamic Inbound Call Routing
    dynamic_inbound_routing_enabled: bool = False

    # Post-Call Webhook
    post_call_webhook_url: str | None = None
    post_call_webhook_headers: Dict[str, str] | None = None
    post_call_webhook_query_params: List[Dict[str, str]] | None = None
    post_call_webhook_custom_payload_enabled: bool = False
    post_call_webhook_custom_payload_template: Dict[str, Any] | None = None

    # Status Webhook
    status_webhook_enabled: bool = False
    status_webhook_url: str | None = None
    status_webhook_headers: Dict[str, str] | None = None
    status_webhook_query_params: List[Dict[str, str]] | None = None

    @field_validator(
        "pre_inbound_webhook_url", "post_call_webhook_url", "status_webhook_url"
    )
    @classmethod
    def _validate_webhook_url(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not v.startswith("https://"):
            raise ValueError("Webhook URL must use HTTPS")
        # Raises SSRFBlockedError (a ValueError subclass) which Pydantic
        # converts to a validation error message.
        assert_public_url(v)
        return v


class SystemWebhooksSettingsResponse(BaseModel):
    """Response body for the System Webhooks settings endpoint.

    Never echoes back decrypted header values — only whether headers are
    configured for each of the three webhooks (`*_headers_configured`).
    """

    model_config = ConfigDict(from_attributes=True)

    pre_inbound_webhook_url: str | None = None
    pre_inbound_webhook_headers_configured: bool = False
    pre_inbound_webhook_query_params: List[Dict[str, str]] = Field(default_factory=list)
    pre_inbound_webhook_static_metadata: Dict[str, str] = Field(default_factory=dict)

    dynamic_inbound_routing_enabled: bool = False

    post_call_webhook_url: str | None = None
    post_call_webhook_headers_configured: bool = False
    post_call_webhook_query_params: List[Dict[str, str]] = Field(default_factory=list)
    post_call_webhook_custom_payload_enabled: bool = False
    post_call_webhook_custom_payload_template: Dict[str, Any] | None = None

    status_webhook_enabled: bool = False
    status_webhook_url: str | None = None
    status_webhook_headers_configured: bool = False
    status_webhook_query_params: List[Dict[str, str]] = Field(default_factory=list)


class SystemWebhookKindEnum(str, Enum):
    pre_inbound = "pre_inbound"
    post_call = "post_call"
    status = "status"


class SystemWebhookTestRequest(BaseModel):
    """Request body for ``POST /api/v2/flows/{flow_id}/system-webhooks/test``.

    Tests whatever is CURRENTLY SAVED for `webhook_kind` on this flow — there
    is no "test unsaved draft" mechanism.
    """

    model_config = ConfigDict(extra="forbid")

    webhook_kind: SystemWebhookKindEnum


class SystemWebhookTestResult(BaseModel):
    """Response body for the System Webhooks test-delivery endpoint — the
    outcome of one synchronous delivery attempt, not the full DB log row."""

    model_config = ConfigDict(from_attributes=True)

    status: str
    status_code: int | None = None
    response_body: str | None = None
    error: str | None = None
    duration_ms: int | None = None


class SystemWebhookDeliveryOut(BaseModel):
    """A single System Webhook delivery log entry, for the deliveries-list
    endpoint. Omits `tenant_id`/`call_flow_id`/`call_session_id` since the
    URL already scopes to one flow; does not include request headers/params/
    body — `SystemWebhookDeliveryLog` never stores those (may carry secrets)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    webhook_kind: SystemWebhookKindEnum
    event_type: str | None = None
    url: str
    status: str
    status_code: int | None = None
    response_body: str | None = None
    error: str | None = None
    attempt_count: int
    duration_ms: int | None = None
    created_at: datetime


class PaginatedSystemWebhookDeliveries(BaseModel):
    """Response body for ``GET /api/v2/flows/{flow_id}/system-webhooks/deliveries``."""

    items: List[SystemWebhookDeliveryOut]
    total: int
    page: int
    page_size: int


class CallFlowOut(BaseModel):
    """Full flow response including all prompt versions."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: uuid.UUID
    name: str
    direction: str
    agent_id: uuid.UUID = Field(..., serialization_alias="agentId")
    # Full AgentOut shape on detail endpoints; slim AgentRef on list
    agent: Dict[str, Any] | None = None
    welcome_message_type: str | None = Field(
        None, serialization_alias="welcomeMessageType"
    )
    custom_welcome_message: str | None = Field(
        None, serialization_alias="customWelcomeMessage"
    )
    current_prompt_id: uuid.UUID | None = Field(
        None, serialization_alias="currentPromptId"
    )
    prompt_versions: List[PromptVersionOut] = Field(
        default_factory=list, serialization_alias="promptVersions"
    )
    flow_data: Dict[str, Any] | None = Field(None, serialization_alias="flowData")
    settings: Dict[str, Any] | None = None
    knowledge_base_ids: List[str] = Field(
        default_factory=list, serialization_alias="knowledgeBaseIds"
    )
    folder_ids: List[uuid.UUID] = Field(
        default_factory=list, serialization_alias="folderIds"
    )
    public_access: bool = Field(False, serialization_alias="publicAccess")
    status: str = "active"
    created_at: datetime = Field(..., serialization_alias="createdAt")
    updated_at: datetime | None = Field(None, serialization_alias="updatedAt")


class CallFlowListItem(BaseModel):
    """Slim flow item used in the paginated list — no prompt_versions payload."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: uuid.UUID
    name: str
    direction: str
    agent_id: uuid.UUID = Field(..., serialization_alias="agentId")
    agent: AgentRef | None = None
    welcome_message_type: str | None = Field(
        None, serialization_alias="welcomeMessageType"
    )
    custom_welcome_message: str | None = Field(
        None, serialization_alias="customWelcomeMessage"
    )
    current_prompt_id: uuid.UUID | None = Field(
        None, serialization_alias="currentPromptId"
    )
    flow_data: Dict[str, Any] | None = Field(None, serialization_alias="flowData")
    settings: Dict[str, Any] | None = None
    knowledge_base_ids: List[str] = Field(
        default_factory=list, serialization_alias="knowledgeBaseIds"
    )
    folder_ids: List[uuid.UUID] = Field(
        default_factory=list, serialization_alias="folderIds"
    )
    public_access: bool = Field(False, serialization_alias="publicAccess")
    status: str = "active"
    created_at: datetime = Field(..., serialization_alias="createdAt")
    updated_at: datetime | None = Field(None, serialization_alias="updatedAt")


class CallFlowListResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    data: List[CallFlowListItem]
    total: int
    page: int
    page_size: int = Field(..., serialization_alias="pageSize")


class FlowValidationError(BaseModel):
    """A single validation failure for a visual flow graph."""

    model_config = ConfigDict(populate_by_name=True)

    code: str
    message: str
    node_id: str | None = Field(None, serialization_alias="nodeId")


class FlowDataUpdate(BaseModel):
    """Request body for ``PUT /api/v2/flows/{flow_id}/flow-data``."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    flow_data: FlowDataSchema = Field(..., alias="flowData")


class FlowDataResponse(BaseModel):
    """Response body for the flow-data GET/PUT endpoints."""

    model_config = ConfigDict(populate_by_name=True)

    flow_data: Dict[str, Any] | None = Field(None, serialization_alias="flowData")
    flow_data_compiled: Dict[str, Any] | None = Field(
        None, serialization_alias="flowDataCompiled"
    )
    validation_errors: List[FlowValidationError] = Field(
        default_factory=list, serialization_alias="validationErrors"
    )


class FlowValidationErrorItem(BaseModel):
    """A single node-level error, per the ticket's literal ``/validate`` shape."""

    node_id: str | None = None
    message: str


class FlowValidationResponse(BaseModel):
    """Response body for ``POST /api/v2/flows/{flow_id}/validate``.

    Ticket-literal shape: ``{valid: bool, errors: [{node_id, message}]}`` — no
    camelCase aliasing, no ``code`` field, deliberately narrower than
    ``FlowValidationError``/``FlowDataResponse`` used elsewhere in this file.
    """

    valid: bool
    errors: List[FlowValidationErrorItem] = Field(default_factory=list)


class FlowDataSaveResponse(BaseModel):
    """Response body for ``PUT /api/v2/flows/{flow_id}/flow-data``.

    Ticket-literal shape: ``{version: int, validated: true}``, optionally including
    ``flowData`` and ``flowDataCompiled`` for compatibility with frontends expecting
    the full graph upon save.
    """

    model_config = ConfigDict(populate_by_name=True)

    version: int
    validated: bool = True
    flow_data: Dict[str, Any] | None = Field(None, serialization_alias="flowData")
    flow_data_compiled: Dict[str, Any] | None = Field(
        None, serialization_alias="flowDataCompiled"
    )


class FlowDataListItem(BaseModel):
    """A single flow's visual/pre-compiled graph, used in the paginated list."""

    model_config = ConfigDict(populate_by_name=True)

    flow_id: uuid.UUID = Field(..., serialization_alias="flowId")
    name: str
    flow_data: Dict[str, Any] | None = Field(None, serialization_alias="flowData")
    flow_data_compiled: Dict[str, Any] | None = Field(
        None, serialization_alias="flowDataCompiled"
    )
    updated_at: datetime | None = Field(None, serialization_alias="updatedAt")


class PaginatedFlowDataResponse(BaseModel):
    """Response body for ``GET /api/v2/flows/flow-data``."""

    model_config = ConfigDict(populate_by_name=True)

    data: List[FlowDataListItem]
    total: int
    page: int
    page_size: int = Field(..., serialization_alias="pageSize")
