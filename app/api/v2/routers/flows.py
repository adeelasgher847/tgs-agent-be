"""
v2 A/B prompt testing + cross-session caller memory endpoints.

Auth: any authenticated tenant principal (API key or JWT); config-rank required
to mutate A/B settings, read-only rank is sufficient to view results. Caller
memory settings require admin rank (owner-equivalent — see require_admin_or_api_key).

PUT  /api/v2/flows/{flow_id}/ab-test
GET  /api/v2/flows/{flow_id}/ab-results
PUT  /api/v2/flows/{flow_id}/ab-test/winner
PUT  /api/v2/flows/{flow_id}/caller-memory-settings
PUT  /api/v2/flows/{flow_id}/post-call-actions-settings
GET  /api/v2/flows/{flow_id}/post-call-actions-settings
PUT  /api/v2/flows/{flow_id}/system-webhooks-settings
GET  /api/v2/flows/{flow_id}/system-webhooks-settings
POST /api/v2/flows/{flow_id}/system-webhooks/test
GET  /api/v2/flows/{flow_id}/system-webhooks/deliveries

Visual Flow Editor endpoints (flow-data, flow-data/validate) live in
app.api.v2.routers.flow_data — a separate router under the same prefix.

Note: the caller memory settings path is deliberately NOT `/{flow_id}/settings` —
that path is already registered by app.api.v2.routers.hipaa for the HIPAA
compliance toggle, and reusing it here would silently shadow that endpoint.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query, Request, status
from sqlalchemy.orm import Session

from app.api.deps import (
    get_db,
    require_admin_or_api_key,
    require_config_or_api_key,
    require_readonly_or_api_key,
)
from app.core.request_auth import ApiKeyPrincipal
from app.models.user import User
from app.schemas.ab_testing import (
    AbResultsResponse,
    AbTestResponse,
    AbTestUpdate,
    AbTestWinnerUpdate,
)
from app.schemas.call_flow import (
    CallerMemorySettingsResponse,
    CallerMemorySettingsUpdate,
    CallScreeningSettingsResponse,
    CallScreeningSettingsUpdate,
    IVRDTMFSettingsResponse,
    IVRDTMFSettingsUpdate,
    MetadataSettingsResponse,
    MetadataSettingsUpdate,
    PaginatedSystemWebhookDeliveries,
    PostCallActionsSettingsResponse,
    PostCallActionsSettingsUpdate,
    SystemWebhookKindEnum,
    SystemWebhooksSettingsResponse,
    SystemWebhooksSettingsUpdate,
    SystemWebhookTestRequest,
    SystemWebhookTestResult,
    VoicemailSettingsResponse,
    VoicemailSettingsUpdate,
)
from app.services.audit_service import log_audit_event
from app.services.call_flow_service import call_flow_service
from app.services.system_webhook_service import run_webhook_test

router = APIRouter(prefix="/flows", tags=["A/B Prompt Testing"])


def _tenant_id(principal: User | ApiKeyPrincipal) -> uuid.UUID:
    return principal.current_tenant_id


@router.put(
    "/{flow_id}/ab-test",
    response_model=AbTestResponse,
    status_code=status.HTTP_200_OK,
    summary="Configure A/B prompt testing on a call flow",
)
def update_ab_test(
    flow_id: uuid.UUID,
    body: AbTestUpdate,
    principal: User | ApiKeyPrincipal = Depends(require_config_or_api_key),
    db: Session = Depends(get_db),
) -> AbTestResponse:
    return call_flow_service.update_ab_test(db, flow_id, _tenant_id(principal), body)


@router.get(
    "/{flow_id}/ab-results",
    response_model=AbResultsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get A/B prompt test results and statistical significance",
)
def get_ab_results(
    flow_id: uuid.UUID,
    principal: User | ApiKeyPrincipal = Depends(require_readonly_or_api_key),
    db: Session = Depends(get_db),
) -> AbResultsResponse:
    return call_flow_service.get_ab_results(db, flow_id, _tenant_id(principal))


@router.put(
    "/{flow_id}/ab-test/winner",
    status_code=status.HTTP_200_OK,
    summary="Promote the winning A/B variant to the flow's active prompt",
)
def promote_ab_winner(
    flow_id: uuid.UUID,
    body: AbTestWinnerUpdate,
    principal: User | ApiKeyPrincipal = Depends(require_config_or_api_key),
    db: Session = Depends(get_db),
) -> dict:
    return call_flow_service.promote_ab_winner(db, flow_id, _tenant_id(principal), body)


@router.put(
    "/{flow_id}/caller-memory-settings",
    response_model=CallerMemorySettingsResponse,
    status_code=status.HTTP_200_OK,
    summary="Configure cross-session caller memory on a call flow",
)
def update_caller_memory_settings(
    flow_id: uuid.UUID,
    body: CallerMemorySettingsUpdate,
    request: Request,
    principal: User | ApiKeyPrincipal = Depends(require_admin_or_api_key),
    db: Session = Depends(get_db),
) -> CallerMemorySettingsResponse:
    tenant_id = _tenant_id(principal)
    result = call_flow_service.update_caller_memory_settings(
        db, flow_id, tenant_id, body
    )

    log_audit_event(
        db,
        request=request,
        tenant_id=tenant_id,
        action="caller_memory_settings.updated",
        resource_type="call_flow",
        resource_id=flow_id,
        new_value={
            "caller_memory_enabled": result.caller_memory_enabled,
            "caller_memory_window": result.caller_memory_window,
        },
        actor_user_id=principal.id,
    )
    return result


@router.put(
    "/{flow_id}/post-call-actions-settings",
    response_model=PostCallActionsSettingsResponse,
    status_code=status.HTTP_200_OK,
    summary="Configure post-call email/Slack summary actions on a call flow",
    description=(
        'Configures the "Post Call Actions" toggles for this call flow:\n\n'
        "- **Email Summary** (`email_summary_enabled` + `email_summary_recipients`): "
        "sends the call summary and extracted variables to an explicit list of "
        "recipient emails after each completed call.\n"
        "- **Summary to Business Owner** (`summary_to_business_owner_enabled`): sends "
        "the same email to the tenant's business owner (the workspace creator's "
        "account email) — there is no separate address field for this, the recipient "
        "is resolved server-side.\n"
        "- **Slack Summary** (`slack_summary_enabled` + `slack_channel_id`/"
        "`slack_channel_name`): posts the call summary and sentiment to a Slack channel "
        "after each completed call. Requires the workspace to have connected Slack "
        "first (see `/api/v1/integrations/slack`); falls back to the workspace's default "
        "channel if `slack_channel_id` is not set.\n\n"
        "All toggles are independent and may be enabled together; email recipients from "
        "both email sources are deduplicated before sending. The request body always "
        "fully replaces all fields (send the current values for any field you don't "
        "want to change). Sending is fire-and-forget after call completion — this "
        "endpoint only updates the stored configuration; it does not send an email or "
        "Slack message itself. Requires admin rank."
    ),
)
def update_post_call_actions_settings(
    flow_id: uuid.UUID,
    body: PostCallActionsSettingsUpdate,
    request: Request,
    principal: User | ApiKeyPrincipal = Depends(require_admin_or_api_key),
    db: Session = Depends(get_db),
) -> PostCallActionsSettingsResponse:
    tenant_id = _tenant_id(principal)
    result = call_flow_service.update_post_call_actions_settings(
        db, flow_id, tenant_id, body
    )

    log_audit_event(
        db,
        request=request,
        tenant_id=tenant_id,
        action="post_call_actions_settings.updated",
        resource_type="call_flow",
        resource_id=flow_id,
        new_value={
            "email_summary_enabled": result.email_summary_enabled,
            "email_summary_recipients": result.email_summary_recipients,
            "summary_to_business_owner_enabled": result.summary_to_business_owner_enabled,
            "slack_summary_enabled": result.slack_summary_enabled,
            "slack_channel_id": result.slack_channel_id,
            "slack_channel_name": result.slack_channel_name,
        },
        actor_user_id=principal.id,
    )
    return result


@router.get(
    "/{flow_id}/post-call-actions-settings",
    response_model=PostCallActionsSettingsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get the currently-saved Post-Call Actions settings for a call flow",
    description=(
        "Returns the currently-saved Post-Call Actions configuration for this "
        "call flow (email summary, summary to business owner, Slack summary). "
        "Read-only rank is sufficient since no secret keys are exposed."
    ),
)
def get_post_call_actions_settings(
    flow_id: uuid.UUID,
    principal: User | ApiKeyPrincipal = Depends(require_readonly_or_api_key),
    db: Session = Depends(get_db),
) -> PostCallActionsSettingsResponse:
    return call_flow_service.get_post_call_actions_settings(
        db, flow_id, _tenant_id(principal)
    )


@router.put(
    "/{flow_id}/voicemail-settings",
    response_model=VoicemailSettingsResponse,
    status_code=status.HTTP_200_OK,
    summary="Configure voicemail detection settings on a call flow",
    description=(
        "Configures the voicemail detection settings for this call flow:\n\n"
        "- **Voicemail Detection** (`voicemail_detection_enabled`): whether voicemail "
        "detection is active on calls using this flow.\n"
        "- **Voicemail Action** (`voicemail_action`): behavior when a voicemail is detected "
        "(`hang_up`, `leave_message`, or `continue`).\n"
        "- **Voicemail Message** (`voicemail_message`): optional message to leave when action "
        "is `leave_message` (up to 500 characters).\n"
        "- **Advanced Detection** (`voicemail_advanced_detection_enabled`): whether Twilio AMD "
        "(Answering Machine Detection) is enabled.\n"
        "- **Detection Timeout** (`voicemail_detection_timeout`): detection timeout in seconds "
        "(1 to 30 seconds).\n\n"
        "Requires admin rank."
    ),
)
def update_voicemail_settings(
    flow_id: uuid.UUID,
    body: VoicemailSettingsUpdate,
    request: Request,
    principal: User | ApiKeyPrincipal = Depends(require_admin_or_api_key),
    db: Session = Depends(get_db),
) -> VoicemailSettingsResponse:
    tenant_id = _tenant_id(principal)
    result = call_flow_service.update_voicemail_settings(
        db, flow_id, tenant_id, body
    )

    log_audit_event(
        db,
        request=request,
        tenant_id=tenant_id,
        action="voicemail_settings.updated",
        resource_type="call_flow",
        resource_id=flow_id,
        new_value={
            "voicemail_detection_enabled": result.voicemail_detection_enabled,
            "voicemail_action": result.voicemail_action,
            "voicemail_message": result.voicemail_message,
            "voicemail_advanced_detection_enabled": (
                result.voicemail_advanced_detection_enabled
            ),
            "voicemail_detection_timeout": result.voicemail_detection_timeout,
        },
        actor_user_id=principal.id,
    )
    return result


@router.get(
    "/{flow_id}/voicemail-settings",
    response_model=VoicemailSettingsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get the currently-saved Voicemail Detection settings for a call flow",
    description=(
        "Returns the currently-saved Voicemail Detection configuration for this call flow "
        "(detection toggle, action, custom message, advanced detection AMD toggle, and timeout). "
        "Read-only rank is sufficient since no secret keys are exposed."
    ),
)
def get_voicemail_settings(
    flow_id: uuid.UUID,
    principal: User | ApiKeyPrincipal = Depends(require_readonly_or_api_key),
    db: Session = Depends(get_db),
) -> VoicemailSettingsResponse:
    return call_flow_service.get_voicemail_settings(
        db, flow_id, _tenant_id(principal)
    )


@router.put(
    "/{flow_id}/call-screening-settings",
    response_model=CallScreeningSettingsResponse,
    status_code=status.HTTP_200_OK,
    summary="Configure Call Screening Detection settings on a call flow",
    description=(
        "Configures the action to take when an automated call screener "
        "(Google Call Screen, Apple/iOS Call Screen, Samsung Smart Call, IVR screeners) "
        "answers an outbound call.\n\n"
        "- **Action** (`call_screening_action`): `respond` (agent states who is calling and why) "
        "or `hang_up` (disconnect immediately).\n\n"
        "Requires admin rank."
    ),
)
def update_call_screening_settings(
    flow_id: uuid.UUID,
    body: CallScreeningSettingsUpdate,
    request: Request,
    principal: User | ApiKeyPrincipal = Depends(require_admin_or_api_key),
    db: Session = Depends(get_db),
) -> CallScreeningSettingsResponse:
    tenant_id = _tenant_id(principal)
    result = call_flow_service.update_call_screening_settings(
        db, flow_id, tenant_id, body
    )

    log_audit_event(
        db,
        request=request,
        tenant_id=tenant_id,
        action="call_screening_settings.updated",
        resource_type="call_flow",
        resource_id=flow_id,
        new_value={
            "call_screening_action": result.call_screening_action,
        },
        actor_user_id=principal.id,
    )
    return result


@router.get(
    "/{flow_id}/call-screening-settings",
    response_model=CallScreeningSettingsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get the currently-saved Call Screening Detection settings for a call flow",
    description=(
        "Returns the currently-saved Call Screening configuration for this call flow "
        "(action: respond or hang_up). "
        "Read-only rank is sufficient since no secret keys are exposed."
    ),
)
def get_call_screening_settings(
    flow_id: uuid.UUID,
    principal: User | ApiKeyPrincipal = Depends(require_readonly_or_api_key),
    db: Session = Depends(get_db),
) -> CallScreeningSettingsResponse:
    return call_flow_service.get_call_screening_settings(
        db, flow_id, _tenant_id(principal)
    )


@router.put(
    "/{flow_id}/metadata-settings",
    response_model=MetadataSettingsResponse,
    status_code=status.HTTP_200_OK,
    summary="Configure Disable Metadata settings on a call flow",
    description=(
        "Configures whether to strip metadata, call_metadata, and metadata query parameters "
        "from outbound API and system webhook payloads.\n\n"
        "- **Disable Metadata** (`disable_metadata`): boolean toggle.\n\n"
        "Requires admin rank."
    ),
)
def update_metadata_settings(
    flow_id: uuid.UUID,
    body: MetadataSettingsUpdate,
    request: Request,
    principal: User | ApiKeyPrincipal = Depends(require_admin_or_api_key),
    db: Session = Depends(get_db),
) -> MetadataSettingsResponse:
    tenant_id = _tenant_id(principal)
    result = call_flow_service.update_metadata_settings(
        db, flow_id, tenant_id, body
    )

    log_audit_event(
        db,
        request=request,
        tenant_id=tenant_id,
        action="metadata_settings.updated",
        resource_type="call_flow",
        resource_id=flow_id,
        new_value={
            "disable_metadata": result.disable_metadata,
        },
        actor_user_id=principal.id,
    )
    return result


@router.get(
    "/{flow_id}/metadata-settings",
    response_model=MetadataSettingsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get the currently-saved Disable Metadata settings for a call flow",
    description=(
        "Returns the currently-saved Disable Metadata configuration for this call flow. "
        "Read-only rank is sufficient since no secret keys are exposed."
    ),
)
def get_metadata_settings(
    flow_id: uuid.UUID,
    principal: User | ApiKeyPrincipal = Depends(require_readonly_or_api_key),
    db: Session = Depends(get_db),
) -> MetadataSettingsResponse:
    return call_flow_service.get_metadata_settings(
        db, flow_id, _tenant_id(principal)
    )


@router.put(
    "/{flow_id}/ivr-dtmf-settings",
    response_model=IVRDTMFSettingsResponse,
    status_code=status.HTTP_200_OK,
    summary="Configure IVR Phone Tree and DTMF Keypad settings on a call flow",
    description=(
        "Configures IVR navigation and real-time DTMF keypad input processing for this call flow.\n\n"
        "- **IVR Navigation** (`ivr_enabled`, `ivr_action`, `ivr_navigation_mode`, `ivr_max_attempts`, "
        "`ivr_keypress_delay`, `ivr_priority_list`, `ivr_wait_on_hold`, `ivr_max_hold_time`)\n"
        "- **DTMF Keypad** (`dtmf_enabled`, `dtmf_button_press_delay`, `dtmf_allow_caller_interruption`, "
        "`dtmf_max_digits`, `dtmf_allowed_exceeded_attempts`, `dtmf_exceeded_action`, `dtmf_end_call_message`)\n\n"
        "Requires admin rank."
    ),
)
def update_ivr_dtmf_settings(
    flow_id: uuid.UUID,
    body: IVRDTMFSettingsUpdate,
    request: Request,
    principal: User | ApiKeyPrincipal = Depends(require_admin_or_api_key),
    db: Session = Depends(get_db),
) -> IVRDTMFSettingsResponse:
    tenant_id = _tenant_id(principal)
    result = call_flow_service.update_ivr_dtmf_settings(
        db, flow_id, tenant_id, body
    )

    log_audit_event(
        db,
        request=request,
        tenant_id=tenant_id,
        action="ivr_dtmf_settings.updated",
        resource_type="call_flow",
        resource_id=flow_id,
        new_value=result.model_dump(),
        actor_user_id=principal.id,
    )
    return result


@router.get(
    "/{flow_id}/ivr-dtmf-settings",
    response_model=IVRDTMFSettingsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get the currently-saved IVR and DTMF settings for a call flow",
    description=(
        "Returns the currently-saved IVR Phone Tree navigation and DTMF keypad settings for this call flow. "
        "Read-only rank is sufficient since no secret keys are exposed."
    ),
)
def get_ivr_dtmf_settings(
    flow_id: uuid.UUID,
    principal: User | ApiKeyPrincipal = Depends(require_readonly_or_api_key),
    db: Session = Depends(get_db),
) -> IVRDTMFSettingsResponse:
    return call_flow_service.get_ivr_dtmf_settings(
        db, flow_id, _tenant_id(principal)
    )


@router.put(
    "/{flow_id}/system-webhooks-settings",
    response_model=SystemWebhooksSettingsResponse,
    status_code=status.HTTP_200_OK,
    summary="Configure System Webhooks (pre-inbound, dynamic routing, post-call, status) on a call flow",
    description=(
        "Configures all four System Webhooks sub-features for this call flow in one PUT:\n\n"
        "- **Pre-Inbound Call Webhook** — fired before an inbound call connects; its "
        '`{"variables": {...}}` response gets injected into the agent\'s prompt/greeting '
        "via `{{key}}` placeholders. Fails open (non-2xx/timeout never blocks the call).\n"
        "- **Dynamic Inbound Call Routing** — depends on the above; if enabled and the "
        "webhook response includes a valid `variables.agent_id`, the call routes to that "
        "agent instead of the number's default.\n"
        "- **Post-Call Webhook** — fired after a call ends with `{callId, agentId, "
        "timestamp, data}`, or a tenant-defined custom JSON payload using `{{field}}` tokens.\n"
        "- **Status Webhook** — fired on call lifecycle sub-events (connect, transfer, end).\n\n"
        "Every field is a full-replace EXCEPT the three `*_headers` fields: since header "
        "values are never echoed back in plaintext, omitting a headers field (`null`) leaves "
        "the stored value unchanged; send `{}` explicitly to clear it. Requires admin rank — "
        "headers may carry auth secrets and routing changes are security-sensitive."
    ),
)
def update_system_webhooks_settings(
    flow_id: uuid.UUID,
    body: SystemWebhooksSettingsUpdate,
    request: Request,
    principal: User | ApiKeyPrincipal = Depends(require_admin_or_api_key),
    db: Session = Depends(get_db),
) -> SystemWebhooksSettingsResponse:
    tenant_id = _tenant_id(principal)
    result = call_flow_service.update_system_webhooks_settings(
        db, flow_id, tenant_id, body
    )

    log_audit_event(
        db,
        request=request,
        tenant_id=tenant_id,
        action="system_webhooks_settings.updated",
        resource_type="call_flow",
        resource_id=flow_id,
        # Deliberately excludes header dicts (secrets) and query-param VALUES
        # (may carry tenant-supplied, potentially sensitive template
        # expressions) — only URLs, toggles, header-presence booleans, and
        # query-param KEYS are logged.
        new_value={
            "pre_inbound_webhook_url": result.pre_inbound_webhook_url,
            "pre_inbound_webhook_headers_configured": (
                result.pre_inbound_webhook_headers_configured
            ),
            "pre_inbound_webhook_query_param_keys": [
                p.get("key") for p in result.pre_inbound_webhook_query_params
            ],
            "dynamic_inbound_routing_enabled": result.dynamic_inbound_routing_enabled,
            "post_call_webhook_url": result.post_call_webhook_url,
            "post_call_webhook_headers_configured": result.post_call_webhook_headers_configured,
            "post_call_webhook_query_param_keys": [
                p.get("key") for p in result.post_call_webhook_query_params
            ],
            "post_call_webhook_custom_payload_enabled": (
                result.post_call_webhook_custom_payload_enabled
            ),
            "status_webhook_enabled": result.status_webhook_enabled,
            "status_webhook_url": result.status_webhook_url,
            "status_webhook_headers_configured": result.status_webhook_headers_configured,
            "status_webhook_query_param_keys": [
                p.get("key") for p in result.status_webhook_query_params
            ],
        },
        actor_user_id=principal.id,
    )
    return result


@router.get(
    "/{flow_id}/system-webhooks-settings",
    response_model=SystemWebhooksSettingsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get the currently-saved System Webhooks settings for a call flow",
    description=(
        "Returns the currently-saved System Webhooks configuration for this call "
        "flow, in the same shape the PUT endpoint returns after a save — for "
        "loading the settings form on page load. Never echoes back decrypted "
        "header values, matching the PUT response; only whether headers are "
        "configured (`*_headers_configured`). Read-only rank is sufficient since "
        "no secrets are exposed either way."
    ),
)
def get_system_webhooks_settings(
    flow_id: uuid.UUID,
    principal: User | ApiKeyPrincipal = Depends(require_readonly_or_api_key),
    db: Session = Depends(get_db),
) -> SystemWebhooksSettingsResponse:
    return call_flow_service.get_system_webhooks_settings(
        db, flow_id, _tenant_id(principal)
    )


@router.post(
    "/{flow_id}/system-webhooks/test",
    response_model=SystemWebhookTestResult,
    status_code=status.HTTP_200_OK,
    summary="Send a one-shot test delivery for a saved System Webhook",
    description=(
        "Fires a single synchronous delivery attempt against whatever is CURRENTLY SAVED "
        "for the given `webhook_kind` on this call flow (save your changes first — there is "
        "no mechanism to test an unsaved draft). Requires admin rank, since viewing a test "
        "result can reveal whether headers/auth are configured correctly. Always records a "
        "`SystemWebhookDeliveryLog` row, same as real call-time deliveries."
    ),
)
async def test_system_webhook(
    flow_id: uuid.UUID,
    body: SystemWebhookTestRequest,
    principal: User | ApiKeyPrincipal = Depends(require_admin_or_api_key),
    db: Session = Depends(get_db),
) -> SystemWebhookTestResult:
    tenant_id = _tenant_id(principal)
    log = await run_webhook_test(db, flow_id, tenant_id, body.webhook_kind.value)
    return SystemWebhookTestResult(
        status=log.status,
        status_code=log.status_code,
        response_body=log.response_body,
        error=log.error,
        duration_ms=log.duration_ms,
    )


@router.get(
    "/{flow_id}/system-webhooks/deliveries",
    response_model=PaginatedSystemWebhookDeliveries,
    status_code=status.HTTP_200_OK,
    summary="List System Webhook delivery history for a call flow",
    description=(
        "Paginated delivery log for this call flow's System Webhooks (pre_inbound, "
        "post_call, status), most recent first. Optionally filter by `webhook_kind`. "
        "Requires admin rank — delivery history can reveal response bodies from the "
        "tenant's own endpoint, the same sensitivity level as the test-delivery "
        "endpoint."
    ),
)
def list_system_webhook_deliveries(
    flow_id: uuid.UUID,
    webhook_kind: SystemWebhookKindEnum | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100, alias="pageSize"),
    principal: User | ApiKeyPrincipal = Depends(require_admin_or_api_key),
    db: Session = Depends(get_db),
) -> PaginatedSystemWebhookDeliveries:
    return call_flow_service.list_system_webhook_deliveries(
        db,
        flow_id,
        _tenant_id(principal),
        webhook_kind.value if webhook_kind is not None else None,
        page,
        page_size,
    )
