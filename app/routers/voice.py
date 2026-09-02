from fastapi import (
    APIRouter,
    BackgroundTasks,
    Request,
    HTTPException,
    Query,
    Depends,
    status,
)
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload
from twilio.twiml.voice_response import VoiceResponse
from datetime import datetime, timezone
import uuid
import requests
import asyncio

from app.core.agent_runtime import resolve_tts_runtime
from app.core.logger import logger
from app.api.deps import get_db, require_tenant, get_optional_tenant_user
from app.schemas.twilio import CallInitiateRequest, CallInitiateResponse
from app.schemas.base import SuccessResponse
from app.services.twilio_service import twilio_service
from app.services.agent_service import agent_service
from app.models.agent import Agent
from app.models.user import User
from app.models.call_session import CallSession
from app.models.call_flow import CallFlow
from app.models.phone_number import PhoneNumber
from app.services.call_flow_service import call_flow_service
from app.services.call_session_service import call_session_service
from app.services.inbound_rules_service import inbound_rules_service
from app.services.voice_screening_qualification_service import (
    maybe_update_resume_status_on_call_completed,
)
from app.services.voice_logging_service import VoiceLoggingService
from app.utils.twilio_validation import (
    validate_twilio_signature,
    validate_twilio_signature_with_token,
    get_request_body,
)
from app.utils.response import create_success_response
from app.core.config import settings
from app.routers.general_websocket import (
    broadcast_call_status_update,
    broadcast_call_ended,
    broadcast_system_notification,
)
from app.services.model_service import ModelService
from app.services.credit_service import credit_service
from app.services.batch_call_completion_service import notify_batch_call_ended
from urllib.parse import quote, urlparse
import re
from app.routers.bidirectional_stream import build_streaming_twiml
from app.utils.voice_twilio_utils import (
    get_twilio_credentials_for_call,
    twilio_caller_id_for_transfer_dial,
)
from app.services.voice_phrase_service import (
    get_random_didnt_catch_response,
)
from app.services.voice_conversation_service import (
    add_to_transcript,
)
from app.services.voice_analysis_service import voice_analysis_service
from app.middleware.request_id_middleware import get_request_id
from app.services.voice_call_service import initiate_call as initiate_call_service
from app.services.voice_analytics_service import voice_analytics_service

router = APIRouter()

# Initialize services
model_service = ModelService()


@router.post("/call/initiate", response_model=SuccessResponse[CallInitiateResponse])
@router.post(
    "/call/initiate/send", response_model=SuccessResponse[CallInitiateResponse]
)
async def initiate_call(
    call_request: CallInitiateRequest,
    http_request: Request,
    user: User | None = Depends(get_optional_tenant_user),
    db: Session = Depends(get_db),
):
    """
    Initiate an outbound voice call.
    /call/initiate/send is an alias for backward compatibility and direct dispatch.

    Auth is resolved here before delegating to the service:
    - JWT user present → standard user-initiated call.
    - No JWT but valid N8N webhook secret → system/webhook call with
      tenant_id resolved from the request body.
    """
    from app.utils.n8n_webhook_verification import verify_n8n_webhook_secret_async

    rid = get_request_id(http_request)

    if user is not None:
        return await initiate_call_service(
            call_request=call_request,
            db=db,
            is_system_call=False,
            tenant_id=user.current_tenant_id,
            user_id=user.id,
            request_id=rid,
        )

    # No JWT user — try webhook secret path
    is_webhook = await verify_n8n_webhook_secret_async(http_request)
    if not is_webhook:
        from fastapi import HTTPException as _HTTPException

        raise _HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: JWT token or n8n webhook secret",
        )

    if not call_request.tenant_id:
        from fastapi import HTTPException as _HTTPException

        raise _HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="tenant_id is required in request body when using webhook secret",
        )
    try:
        tenant_uuid = uuid.UUID(call_request.tenant_id)
        user_uuid = uuid.UUID(call_request.user_id) if call_request.user_id else None
    except ValueError:
        from fastapi import HTTPException as _HTTPException

        raise _HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid UUID format for tenant_id or user_id",
        )

    return await initiate_call_service(
        call_request=call_request,
        db=db,
        is_system_call=True,
        tenant_id=tenant_uuid,
        user_id=user_uuid,
        request_id=rid,
    )


def _resolve_default_inbound_call_flow(
    db: Session, agent_id: uuid.UUID, tenant_id: uuid.UUID
) -> CallFlow | None:
    """Resolve the default inbound `CallFlow` for a given agent/tenant, used by
    both the baseline resolution and the Dynamic Inbound Routing override
    lookup in `handle_incoming_call`. Returns None (not an error) when no
    active inbound/bidirectional call flow exists — the pre-existing
    no-call-flow behavior is preserved in that case.
    """
    return (
        db.execute(
            select(CallFlow)
            .where(
                CallFlow.tenant_id == tenant_id,
                CallFlow.agent_id == agent_id,
                CallFlow.direction.in_(["inbound", "bidirectional"]),
                CallFlow.status == "active",
                CallFlow.is_deleted.is_(False),
            )
            .order_by(
                CallFlow.updated_at.desc().nullslast(), CallFlow.created_at.desc()
            )
            .limit(1)
        )
        .scalars()
        .first()
    )


@router.post("/incoming", response_class=HTMLResponse, include_in_schema=False)
async def handle_incoming_call(
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Twilio inbound voice webhook entrypoint.
    Resolves tenant by called number, routes to tenant's dedicated inbound agent,
    creates an inbound call session, and returns Connect/Stream TwiML.
    """

    def _fallback_twiml(message: str) -> HTMLResponse:
        response = VoiceResponse()
        response.say(message)
        response.hangup()
        return HTMLResponse(str(response), media_type="application/xml")

    try:
        form_data = await request.form()
        call_sid = form_data.get("CallSid", "")
        from_number = form_data.get("From", "")
        to_number = form_data.get("To", "")

        if not to_number:
            logger.warning("Inbound webhook missing 'To' number")
            return _fallback_twiml(
                "Sorry, we could not identify the destination number for this call."
            )

        phone_number = (
            db.query(PhoneNumber)
            .filter(
                PhoneNumber.phone_number == to_number,
                PhoneNumber.status == "active",
            )
            .first()
        )
        if not phone_number:
            logger.warning("Inbound number not assigned: %s", to_number)
            return _fallback_twiml(
                "Sorry, this number is not configured for inbound service."
            )

        if not settings.ALLOW_UNAUTHENTICATED_WEBHOOKS:
            is_valid_signature = False
            # Twilio signs form params as a dict — pass parsed fields, not raw body.
            form_params = dict(form_data)

            # Multi-account support: validate with number-specific token when available.
            if phone_number.twilio_auth_token:
                try:
                    from app.core.security import decrypt_api_key

                    number_auth_token = decrypt_api_key(phone_number.twilio_auth_token)
                    is_valid_signature = validate_twilio_signature_with_token(
                        request, form_params, number_auth_token
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to validate inbound signature with number-specific token "
                        "(tenant_id=%s, number=%s, call_sid=%s): %s",
                        phone_number.tenant_id,
                        to_number,
                        call_sid,
                        e,
                    )

            # Backward-compatible fallback to global token.
            if not is_valid_signature:
                is_valid_signature = validate_twilio_signature(request, form_params)

            if not is_valid_signature:
                logger.warning(
                    "Inbound signature validation failed (tenant_id=%s, number=%s, call_sid=%s)",
                    phone_number.tenant_id,
                    to_number,
                    call_sid,
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid Twilio signature",
                )

        # Resolve target agent: Check if a specific assistant/agent is linked directly to this PhoneNumber
        inbound_agent = None
        if phone_number.assistant_id:
            inbound_agent = (
                db.query(Agent)
                .filter(
                    Agent.id == phone_number.assistant_id,
                    Agent.tenant_id == phone_number.tenant_id,
                    ~Agent.is_deleted,
                )
                .first()
            )

        if not inbound_agent:
            logger.warning(
                "Inbound call rejected: No active agent linked to number %s",
                to_number,
            )
            return _fallback_twiml(
                "Sorry, this number is not configured to receive calls."
            )

        # System Webhooks — Pre-Inbound Call Webhook + Dynamic Inbound Call Routing.
        # Best-effort: any failure here must fall back to today's behavior
        # (inbound_agent / no call flow), never break inbound call routing.
        resolved_agent = inbound_agent
        resolved_flow = None
        webhook_variables: dict = {}
        try:
            default_call_flow = _resolve_default_inbound_call_flow(
                db, inbound_agent.id, phone_number.tenant_id
            )
            resolved_flow = default_call_flow

            if default_call_flow and default_call_flow.pre_inbound_webhook_url:
                from app.services.system_webhook_service import (
                    fetch_pre_inbound_webhook_variables,
                )

                webhook_variables = await fetch_pre_inbound_webhook_variables(
                    db,
                    default_call_flow,
                    from_number=from_number,
                    to_number=to_number,
                )

            if (
                default_call_flow
                and default_call_flow.dynamic_inbound_routing_enabled
                and webhook_variables.get("agent_id")
            ):
                try:
                    override_agent_id = uuid.UUID(webhook_variables["agent_id"])
                except (ValueError, TypeError, AttributeError):
                    logger.warning(
                        "Dynamic inbound routing: malformed agent_id %r from webhook "
                        "(call_flow=%s) — falling back to default agent",
                        webhook_variables.get("agent_id"),
                        default_call_flow.id,
                    )
                    override_agent_id = None

                override_agent = None
                if override_agent_id is not None:
                    override_agent = db.execute(
                        select(Agent).where(
                            Agent.id == override_agent_id,
                            Agent.tenant_id == phone_number.tenant_id,
                            ~Agent.is_deleted,
                        )
                    ).scalar_one_or_none()
                    if override_agent is None:
                        logger.warning(
                            "Dynamic inbound routing: no active agent %s in tenant %s "
                            "(call_flow=%s) — falling back to default agent",
                            override_agent_id,
                            phone_number.tenant_id,
                            default_call_flow.id,
                        )

                if override_agent is not None:
                    override_flow = _resolve_default_inbound_call_flow(
                        db, override_agent.id, phone_number.tenant_id
                    )
                    resolved_agent = override_agent
                    resolved_flow = (
                        override_flow if override_flow else default_call_flow
                    )
        except (
            Exception
        ) as exc:  # defensive — never let webhook routing break inbound calls
            logger.warning(
                "System webhook inbound routing failed for number %s (tenant=%s): %s",
                to_number,
                phone_number.tenant_id,
                exc,
                exc_info=True,
            )
            resolved_agent = inbound_agent
            resolved_flow = None
            webhook_variables = {}

        target_flow = resolved_flow or default_call_flow

        if resolved_agent is None:
            logger.warning(
                "No agent resolved for inbound call %s on phone number %s",
                call_sid,
                to_number,
            )
            return _fallback_twiml("Sorry, this call cannot be connected at this time.")

        # ── Inbound Rules & Number Blocking Check ──
        if target_flow and target_flow.inbound_rule_set_id:
            is_blocked, matched_rule = (
                inbound_rules_service.is_number_blocked(
                    db=db,
                    tenant_id=phone_number.tenant_id,
                    rule_set_id=target_flow.inbound_rule_set_id,
                    phone_number=from_number,
                )
            )
            if is_blocked:
                logger.info(
                    "Inbound call %s from %s blocked by rule set %s (rule: %s, label: %s)",
                    call_sid,
                    from_number,
                    target_flow.inbound_rule_set_id,
                    matched_rule.id if matched_rule else None,
                    matched_rule.label if matched_rule else None,
                )
                try:
                    # Attribution note: Inbound calls have no authenticated caller user.
                    # Associate with the agent creator user_id if present, or None for API-key/system-created agents.
                    agent_owner_user_id = (
                        getattr(resolved_agent, "created_by", None)
                        if resolved_agent
                        else None
                    )
                    blocked_session = CallSession(
                        tenant_id=phone_number.tenant_id,
                        user_id=agent_owner_user_id,
                        agent_id=resolved_agent.id,
                        call_flow_id=target_flow.id,
                        twilio_call_sid=call_sid,
                        from_number=from_number,
                        to_number=to_number,
                        customer_phone_number=from_number,
                        assistant_phone_number=to_number,
                        call_type="inbound",
                        status="completed",
                        ended_reason="Blocked by inbound rule set",
                        start_time=datetime.now(timezone.utc),
                        end_time=datetime.now(timezone.utc),
                    )
                    db.add(blocked_session)
                    db.commit()
                    db.refresh(blocked_session)
                except Exception as db_err:
                    db.rollback()
                    logger.warning(
                        "Could not save blocked call session (call_sid=%s): %s",
                        call_sid,
                        db_err,
                    )

                response = VoiceResponse()
                response.reject(reason="busy")
                return HTMLResponse(str(response), media_type="application/xml")

        # ── Inbound Call Redirection & Forwarding Check ──
        if (
            target_flow
            and target_flow.redirect_inbound_calls_enabled
            and target_flow.redirect_forward_phone_number
        ):
            static_metadata = dict(
                target_flow.pre_inbound_webhook_static_metadata or {}
            )
            redirect_context = {
                "from": from_number,
                "to": to_number,
                "caller_phone": from_number,
                "caller_number": from_number,
                "From": from_number,
                "To": to_number,
                "_metadata": static_metadata,
                "_variable": dict(webhook_variables or {}),
                **static_metadata,
                **dict(webhook_variables or {}),
            }

            conditions_met = call_flow_service.evaluate_redirect_conditions(
                target_flow.redirect_conditions or [],
                redirect_context,
            )

            if conditions_met:
                logger.info(
                    "Inbound call redirection triggered for call %s (flow=%s, forward_to=%s)",
                    call_sid,
                    target_flow.id,
                    target_flow.redirect_forward_phone_number,
                )
                try:
                    # Attribution note: Inbound calls have no authenticated caller user.
                    # Associate with the agent creator user_id if present, or None for API-key/system-created agents.
                    agent_owner_user_id = (
                        getattr(resolved_agent, "created_by", None)
                        if resolved_agent
                        else None
                    )
                    redirect_session = CallSession(
                        tenant_id=phone_number.tenant_id,
                        user_id=agent_owner_user_id,
                        agent_id=resolved_agent.id,
                        call_flow_id=target_flow.id,
                        twilio_call_sid=call_sid,
                        from_number=from_number,
                        to_number=to_number,
                        customer_phone_number=from_number,
                        assistant_phone_number=to_number,
                        call_type="inbound",
                        status="completed",
                        transferred=True,
                        ended_reason="Inbound call redirected",
                        start_time=datetime.now(timezone.utc),
                        end_time=datetime.now(timezone.utc),
                    )
                    db.add(redirect_session)
                    db.commit()
                    db.refresh(redirect_session)
                except Exception as db_err:
                    db.rollback()
                    logger.warning(
                        "Could not save redirected call session (call_sid=%s): %s",
                        call_sid,
                        db_err,
                    )

                response = VoiceResponse()
                if (
                    target_flow.redirect_speak_message_enabled
                    and target_flow.redirect_message
                ):
                    rendered_msg = (
                        call_flow_service.render_redirect_message_template(
                            target_flow.redirect_message,
                            redirect_context,
                        )
                    )
                    if rendered_msg:
                        response.say(rendered_msg)

                dial = response.dial(
                    caller_id=from_number if from_number else None
                )
                dial.number(target_flow.redirect_forward_phone_number)
                return HTMLResponse(str(response), media_type="application/xml")

        # Billing guardrail: enforce the same credit gating used in outbound flows.
        if not resolved_agent.model:
            logger.warning(
                "Inbound agent %s has no model configured", resolved_agent.id
            )
            return _fallback_twiml(
                "Sorry, this inbound agent is not configured correctly right now."
            )

        model_name = resolved_agent.model.model_name
        inbound_tts_runtime = resolve_tts_runtime(resolved_agent, db=db)
        tts_provider_slug = inbound_tts_runtime.adapter_slug
        has_sufficient, current_credits, required_credits, decline_reason = (
            credit_service.has_sufficient_credits(
                db=db,
                tenant_id=phone_number.tenant_id,
                model_name=model_name,
                tts_provider_slug=tts_provider_slug,
                is_byo_elevenlabs=inbound_tts_runtime.is_byo_elevenlabs,
            )
        )
        if not has_sufficient:
            logger.warning(
                "Inbound credit check failed for tenant %s: current=%s required=%s model=%s reason=%s",
                phone_number.tenant_id,
                current_credits,
                required_credits,
                model_name,
                decline_reason,
            )
            return _fallback_twiml(
                "Sorry, this service is currently unavailable. Please try again later."
            )

        call_session = call_session_service.create_call_session(
            db=db,
            user_id=resolved_agent.created_by,
            agent_id=resolved_agent.id,
            tenant_id=phone_number.tenant_id,
            twilio_call_sid=call_sid,
            from_number=from_number,
            to_number=to_number,
            call_type="inbound",
            assistant_phone_number=to_number,
            customer_phone_number=from_number,
        )

        # Attach the resolved call flow + any webhook-provided variables so both
        # live-call handlers (Twilio / LiveKit) can key off them (call_flow_id
        # for settings, call_metadata.webhook_variables for {{key}} injection).
        # Also detect + link a call-drop reconnect (F-11) via parent_call_id +
        # call_metadata. The dropped-session lookup itself lives inside this
        # same try block so a query failure here fails open exactly like the
        # rest of this block, instead of bubbling up and failing the call.
        try:
            if resolved_flow is not None:
                call_session.call_flow_id = resolved_flow.id
            metadata_updates: dict = {}
            if webhook_variables:
                metadata_updates["webhook_variables"] = webhook_variables
            dropped_session = call_session_service.find_recent_dropped_session(
                db=db,
                from_number=from_number,
                tenant_id=phone_number.tenant_id,
                within_seconds=300,
            )
            if dropped_session is not None:
                call_session.parent_call_id = dropped_session.id
                metadata_updates["is_reconnect"] = True
                metadata_updates["reconnect_from_session_id"] = str(
                    dropped_session.id
                )
                logger.info(
                    "Detected call-drop reconnect: call_session=%s reconnect_from=%s",
                    call_session.id,
                    dropped_session.id,
                )
            if metadata_updates:
                call_session.call_metadata = {
                    **(call_session.call_metadata or {}),
                    **metadata_updates,
                }
            if resolved_flow is not None or webhook_variables or dropped_session is not None:
                db.commit()
        except Exception as exc:  # defensive — never let this break inbound calls
            logger.warning(
                "Failed to attach call_flow/webhook_variables/reconnect data to call_session %s: %s",
                call_session.id,
                exc,
            )
            db.rollback()

        twiml = build_streaming_twiml(
            call_session_id=str(call_session.id),
            agent_id=str(resolved_agent.id),
        )
        return HTMLResponse(twiml, media_type="application/xml")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to handle inbound call webhook: %s", e, exc_info=True)
        return _fallback_twiml("Sorry, we are unable to connect your call right now.")


# Twilio CallStatus → internal lifecycle status (GAP 4).
# "busy" maps to "no_answer" per ticket spec (both mean the callee was unreachable).
# "in-progress" and "answered" are skipped for outbound (handled by first media packet
# in bidirectional_stream.py); they map to "connected" for inbound/other flows.
# Documented Twilio CallStatus values: https://www.twilio.com/docs/voice/api/call-resource#call-status-values
# "queued" (call queued before dialing) maps to "initiated" — same pre-connection
# semantics as the existing "initiated" entry.
# "canceled" (call canceled via REST API before it was answered) maps to "failed" —
# same "never completed" semantics as the existing "failed" entry.
_TWILIO_TO_INTERNAL_STATUS: dict[str, str] = {
    "queued": "initiated",
    "initiated": "initiated",
    "ringing": "ringing",
    "in-progress": "connected",
    "answered": "connected",
    "completed": "completed",
    "failed": "failed",
    "canceled": "failed",
    "no-answer": "no_answer",
    "busy": "no_answer",
}


def _commit_terminal_call_session_status(
    db: Session, call_session: CallSession | None
) -> None:
    """
    Persist the call_session.status mutation applied earlier in
    handle_call_events_webhook (`call_session.status = internal_status`).

    Only the "completed" branch persists status today, via
    call_session_service.update_call_session_status() (which commits). The
    "failed" / "busy" / "no-answer" / "canceled" branches previously only
    mutated the in-memory ORM object with no commit anywhere in the request —
    the transition was silently lost once the request's DB session closed
    without a commit (SessionLocal is autocommit=False/autoflush=False).

    Deliberately NOT a swap to update_call_session_status(): that method has
    additional side effects (end_time/duration, call log update, inbound CRM
    sync scheduling for completed/failed/busy) that aren't a byproduct-free fit
    here and would risk double-firing behavior already handled explicitly by
    this webhook's elif chain (webhook fire, credit-monitoring stop, batch
    completion notify). This is intentionally just the missing commit.
    """
    if call_session is None:
        return
    try:
        db.commit()
    except Exception as e:
        logger.warning(
            "⚠️ Failed to persist call session status update (non-critical): %s", e
        )


@router.post("/call-events", response_class=HTMLResponse, include_in_schema=False)
@router.post(
    "/webhook/call-events", response_class=HTMLResponse, include_in_schema=False
)
async def handle_call_events_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    agentId: str | None = Query(None),
    userId: str | None = Query(None),
    callSessionId: str | None = Query(None),
    timeout: str | None = Query(None),
    body: str = Depends(get_request_body),
    db: Session = Depends(get_db),
):
    logger.info("🔥🔥🔥 WEBHOOK CALLED! 🔥🔥🔥")
    logger.info("=== Call Events Webhook Started ===")
    logger.info("Timestamp: %s", datetime.now(timezone.utc).isoformat())
    from app.core.pii_redactor import prepare_request_log_context

    logger.info(
        "Call events webhook started %s",
        prepare_request_log_context(
            request.method,
            request.url.path,
            request.headers,
            query_params={
                "agentId": agentId or "",
                "userId": userId or "",
                "callSessionId": callSessionId or "",
            },
            body_length=len(body) if body else 0,
        ),
    )

    # Optional WebSocket broadcast (non-blocking - fire and forget)
    try:
        asyncio.create_task(
            broadcast_system_notification(
                notification_type="webhook_started",
                message=f"Webhook started for call session {callSessionId}",
                metadata={
                    "agent_id": agentId,
                    "user_id": userId,
                    "call_session_id": callSessionId,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        )
        logger.info("✅ WebSocket broadcast queued at webhook start")
    except Exception as e:
        logger.warning("⚠️ WebSocket broadcast failed (non-critical): %s", e)
        # Don't print traceback - this is not critical for call processing
    try:
        logger.debug("Parsing request body...")

        # Parse form data to get call information
        form_data = await request.form()
        call_sid = form_data.get("CallSid", "")
        call_status = form_data.get("CallStatus", "")
        from_number = form_data.get("From", "")
        to_number = form_data.get("To", "")
        direction = form_data.get("Direction", "")

        # Note: Speech input is now handled by Deepgram STT via WebSocket
        # The old Twilio SpeechResult is no longer used
        # speech_result = form_data.get("SpeechResult", "")
        # confidence = form_data.get("Confidence", "")
        # speech_duration = form_data.get("SpeechDuration", "")

        logger.info("🎤 Speech handling is now managed by Deepgram STT WebSocket")

        # Get call session using callSessionId first, then fallback to Twilio CallSid.
        call_session = None
        agent = None

        if callSessionId:
            try:
                session_uuid = uuid.UUID(callSessionId)
                call_session = call_session_service.get_call_session_by_id(
                    db, session_uuid
                )
                if call_session:
                    logger.info(
                        "✅ Found call session: %s from query parameter", call_session.id
                    )

                    # Fetch agent using call session's tenant_id
                    if agentId:
                        agent = agent_service.get_agent_by_id(
                            db, uuid.UUID(agentId), call_session.tenant_id
                        )
                        if agent:
                            logger.info(
                                "✅ Agent fetched: %s (ID: %s)", agent.name, agent.id
                            )
                            logger.info("🏢 Tenant: %s", agent.tenant_id)
                        else:
                            logger.warning(
                                "⚠️ Agent %s not found in tenant %s", agentId, call_session.tenant_id
                            )
                else:
                    logger.warning("⚠️ No call session found for ID: %s", callSessionId)
            except ValueError:
                logger.warning("⚠️ Invalid call session ID format: %s", callSessionId)
        else:
            logger.info("⚠️ No callSessionId provided in query parameters")

        # Fallback lookup by Twilio SID for inbound and legacy callback URLs
        if not call_session and call_sid:
            call_session = call_session_service.get_call_session_by_twilio_sid(
                db, call_sid
            )
            if call_session:
                logger.info(
                    "✅ Found call session via CallSid fallback: %s", call_session.id
                )
                if not agent and call_session.agent_id:
                    try:
                        agent = agent_service.get_agent_by_id(
                            db,
                            call_session.agent_id,
                            call_session.tenant_id,
                        )
                    except Exception:
                        agent = None

        # Validate request — Twilio signature only.
        #
        # This endpoint is exclusively used as a Twilio `status_callback` URL (see
        # phone_number_service.py / voice_call_service.py / call_control_mixin.py —
        # every caller that builds this URL does so for a Twilio `calls.create` /
        # `<Dial>` status_callback). There is no browser/WebRTC-originated caller of
        # this endpoint anywhere in the codebase: the Web SDK calling feature
        # (app/routers/sdk.py `public-call-token`) connects browser clients directly
        # to a LiveKit room and never posts here. The previous `elif is_webrtc` branch
        # accepted *any* `Authorization: Bearer <anything>` header via a no-op stub
        # (`validate_webrtc_auth`), which let an attacker bypass X-Twilio-Signature
        # enforcement entirely by sending an arbitrary Authorization header instead —
        # removed rather than wired up, since there's no real auth mechanism to wire
        # it to.
        is_twilio = "X-Twilio-Signature" in request.headers

        if is_twilio:
            form_params = dict(form_data)
            if not await _validate_transfer_webhook_signature(
                request, db, call_session, form_params
            ):
                logger.warning(
                    "Call events webhook: invalid Twilio signature (call_sid=%s, callSessionId=%s)",
                    call_sid,
                    callSessionId,
                )
                raise HTTPException(status_code=403, detail="Invalid Twilio signature")
        else:
            if not settings.ALLOW_UNAUTHENTICATED_WEBHOOKS:
                logger.warning(
                    "Call events webhook: no Twilio signature present and unauthenticated webhooks disallowed"
                )
                raise HTTPException(status_code=403, detail="Missing Twilio signature")
            logger.info("No authentication headers found, allowing for testing")

        # (Removed outbound in-progress gating based on AnsweredBy/has_media)

        # Log the call event
        logger.info(
            "Call Events Webhook - SID: %s, Status: %s, From: %s, To: %s, Direction: %s", call_sid, call_status, from_number, to_number, direction
        )
        logger.info("AgentId from query: %s", agentId)

        # 🔍 DEBUG: Track all incoming statuses for troubleshooting
        logger.debug("=" * 60)
        logger.debug("🔍 DEBUG WEBHOOK RECEIVED:")
        logger.debug("   Status: '%s'", call_status)
        logger.debug("   Direction: '%s'", direction)
        logger.debug("   Call SID: %s", call_sid)
        if call_session:
            logger.debug("   Current DB Status: '%s'", call_session.status)
            logger.debug("   Call Session ID: %s", call_session.id)
        else:
            logger.debug("   Call Session: Not found")
        logger.debug("=" * 60)

        # Test WebSocket connection if we have a call session (non-blocking - fire and forget)
        # if call_session:
        #     try:
        #         asyncio.create_task(broadcast_call_status_update(
        #             call_session_id=str(call_session.id),
        #             status="webhook_test",
        #             metadata={
        #                 "message": "Webhook is working",
        #                 "timestamp": datetime.now(timezone.utc).isoformat(),
        #                 "call_sid": call_sid
        #             }
        #         ))
        #         logger.info(f"✅ Test broadcast queued to WebSocket for session {call_session.id}")
        #     except Exception as e:
        #         logger.warning(f"⚠️ Test broadcast failed (non-critical): {e}")

        # Status broadcasts will be handled in the main status update section below

        # Update call session status if we have a call session and status
        # ⚠️ SKIP automatic update for "answered" and "in-progress" - handled in specific handlers below
        # "in-progress" will ONLY be set when media streaming actually starts (first media packet in bidirectional_stream.py)
        internal_status = _TWILIO_TO_INTERNAL_STATUS.get(call_status, call_status)
        if (
            call_session
            and call_status
            and (
                call_status not in ["answered", "in-progress"] or direction == "inbound"
            )
        ):
            logger.info(
                "🔄 Updating call session %s status: %s → %s",
                call_session.id,
                call_status,
                internal_status,
            )
            _previous_status = call_session.status
            call_session.status = internal_status

            # Status Webhook — "connect" event. Only fire on an actual new
            # transition into "connected" (not on repeated Twilio status
            # callbacks for the same status), and only when a System Webhook
            # is actually configured for this call flow — never block/slow
            # the webhook response on this.
            if internal_status == "connected" and _previous_status != "connected":
                try:
                    _call_flow = (
                        db.query(CallFlow)
                        .filter(
                            CallFlow.id == call_session.call_flow_id,
                            CallFlow.tenant_id == call_session.tenant_id,
                        )
                        .first()
                        if call_session.call_flow_id
                        else None
                    )
                    if _call_flow and _call_flow.status_webhook_enabled:
                        from app.services.system_webhook_service import (
                            schedule_status_webhook,
                        )

                        schedule_status_webhook(call_session.id, "call.connected")
                except Exception as exc:
                    logger.warning(
                        "Status webhook (connect) schedule failed (non-critical): %s",
                        exc,
                    )
        elif call_session and call_status in ["answered", "in-progress"]:
            logger.debug(
                "🔍 DEBUG: Skipping automatic status update for '%s' - will be set when media streaming starts", call_status
            )

        # Set end time and calculate duration when call completes
        if call_session and call_status == "completed":
            call_session.end_time = datetime.now(timezone.utc)
            if call_session.start_time:
                duration = (
                    call_session.end_time - call_session.start_time
                ).total_seconds()
                call_session.duration = int(duration)
                logger.info(
                    "⏰ Set end time and duration (%ss) for session %s", duration, call_session.id
                )

                # Broadcast call ended event (non-blocking - fire and forget)
                try:
                    asyncio.create_task(
                        broadcast_call_ended(
                            call_session_id=str(call_session.id),
                            reason="completed",
                            final_data={
                                "call_sid": call_sid,
                                "from_number": from_number,
                                "to_number": to_number,
                                "direction": direction,
                                "duration": call_session.duration,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                    )
                    logger.info(
                        "✅ Queued call ended event for session %s", call_session.id
                    )
                except Exception as e:
                    logger.warning(
                        "⚠️ Failed to queue call ended event (non-critical): %s", e
                    )

                # Stop credit monitoring when call completes
                try:
                    credit_service.stop_credit_monitoring(call_session.id)
                    logger.info(
                        "✅ Stopped credit monitoring for call session %s", call_session.id
                    )
                except Exception as e:
                    logger.warning(
                        "⚠️ Failed to stop credit monitoring (non-critical): %s", e
                    )

            # Update call session AND call log together (single commit)
            call_session_service.update_call_session_status(
                db, call_session.id, "completed", ended_reason="hung up"
            )
            try:
                maybe_update_resume_status_on_call_completed(db, call_session.id)
            except Exception as mq_exc:
                logger.warning(
                    "Resume screening qualify on completed webhook: %s",
                    mq_exc,
                    exc_info=True,
                )

            try:
                await notify_batch_call_ended(db, call_session.id, call_status)
            except Exception as batch_exc:
                logger.warning(
                    "Batch call completion hook failed: %s", batch_exc, exc_info=True
                )

            # Schedule GCS recording upload (non-blocking; skipped if recording_enabled=false)
            try:
                from app.services.call_recording_upload_service import (
                    schedule_recording_upload,
                )

                schedule_recording_upload(call_session.id)
            except Exception as _ru_exc:
                logger.warning("Recording upload schedule failed: %s", _ru_exc)

            logger.info(
                "✅ Updated call session %s status to: %s with ended_reason: hung up", call_session.id, call_status
            )

            # Broadcast status update to WebSocket (SINGLE COMPREHENSIVE BROADCAST)
            # SKIP "in-progress" status here - it will be sent when media stream starts
            if call_status == "in-progress":
                logger.info(
                    "ℹ️ Skipping 'in-progress' broadcast here - will be sent by media stream handler"
                )
            else:
                try:
                    logger.info(
                        "🚀 Broadcasting call status update: %s for session %s", call_status, call_session.id
                    )

                    # Prepare comprehensive metadata
                    metadata = {
                        # "call_sid": call_sid,
                        "from_number": from_number,
                        "to_number": to_number,
                        "direction": direction,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "start_time": (
                            call_session.start_time.isoformat()
                            if call_session.start_time
                            else None
                        ),
                        "end_time": (
                            call_session.end_time.isoformat()
                            if call_session.end_time
                            else None
                        ),
                        "duration": call_session.duration,
                    }

                    # Add status-specific messages
                    if call_status == "ringing":
                        metadata["message"] = "Call is ringing"
                    elif call_status == "completed":
                        metadata["message"] = "Call has been completed"

                    await broadcast_call_status_update(
                        call_session_id=str(call_session.id),
                        status=call_status,
                        metadata=metadata,
                    )
                    logger.debug(
                        "✅ Call status update sent: %s for session %s", call_status, call_session.id
                    )

                    # Also broadcast call ended event for completed calls (non-blocking - fire and forget)
                    if call_status == "completed":
                        asyncio.create_task(
                            broadcast_call_ended(
                                call_session_id=str(call_session.id),
                                reason="Call completed",
                                final_data={
                                    "call_sid": call_sid,
                                    "duration": call_session.duration,
                                    "end_time": call_session.end_time.isoformat(),
                                    "transcript": call_session.call_transcript or [],
                                },
                            )
                        )
                        logger.debug(
                            "✅ Queued call ended event for session %s", call_session.id
                        )

                except Exception as e:
                    logger.error(
                        "❌ Failed to broadcast call status update: %s", e, exc_info=True
                    )
        else:
            if not call_session:
                logger.warning(
                    "⚠️ No call session found - cannot update status or broadcast"
                )
            if not call_status:
                logger.warning(
                    "⚠️ No call status provided - cannot update status or broadcast"
                )

        # Speech input is now handled by Deepgram STT via WebSocket
        # The WebSocket will transcribe audio and generate responses
        # This webhook now primarily handles call status updates and plays pending responses

        # Handle different call statuses and trigger agent logic
        logger.info(
            "Processing call status: '%s' with direction: '%s'", call_status, direction
        )

        if call_status == "initiated" and direction == "outbound-api":
            # Call has been initiated - just log and return empty response
            logger.info("Call initiated - SID: %s", call_sid)

            # Broadcast call initiated event (non-blocking - fire and forget)
            if call_session:
                try:
                    asyncio.create_task(
                        broadcast_call_status_update(
                            call_session_id=str(call_session.id),
                            status="initiated",
                            metadata={
                                "check": "just checking",
                                "call_sid": call_sid,
                                "direction": direction,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                    )
                    logger.debug(
                        "✅ Broadcasted call initiated event for session %s", call_session.id
                    )
                except Exception as e:
                    logger.error("❌ Failed to broadcast call initiated event: %s", e)

                # Fire call.started webhook
                try:
                    from app.services.webhook_service import fire_webhooks

                    background_tasks.add_task(
                        fire_webhooks,
                        call_session.tenant_id,
                        "call.started",
                        {
                            "call_session_id": str(call_session.id),
                            "call_sid": call_sid,
                            "direction": direction,
                            "from_number": from_number,
                            "to_number": to_number,
                        },
                    )
                except Exception as _wh_exc:
                    logger.warning("call.started webhook fire failed: %s", _wh_exc)

            return HTMLResponse("", media_type="application/xml")

        elif call_status == "ringing" and direction == "outbound-api":
            # Outbound call is ringing - just log, don't play any audio
            logger.info("🔔 CALL IS RINGING - SID: %s", call_sid)

            # Broadcast call ringing event (non-blocking - fire and forget)
            if call_session:
                try:
                    asyncio.create_task(
                        broadcast_call_status_update(
                            call_session_id=str(call_session.id),
                            status="ringing",
                            metadata={
                                "call_sid": call_sid,
                                "direction": direction,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                    )
                    logger.debug(
                        "✅ Broadcasted call ringing event for session %s", call_session.id
                    )
                except Exception as e:
                    logger.error("❌ Failed to broadcast call ringing event: %s", e)

            # Return empty response - no audio should play while ringing
            return HTMLResponse("", media_type="application/xml")

        elif call_status == "answered" and direction == "outbound-api":
            # ⚠️ IGNORE - We use first media packet detection instead (VAPI-style)
            logger.info(
                "ℹ️ ANSWERED STATUS RECEIVED (ignored - using first media packet instead)"
            )
            logger.debug(
                "🔍 DEBUG: Will wait for first media packet from WebSocket stream"
            )
            logger.debug(
                "🔍 DEBUG: User pickup detection happens in bidirectional_stream.py"
            )

            # Don't start credit monitoring or update status here
            # Wait for first media packet event from WebSocket stream

            return HTMLResponse("", media_type="application/xml")

        elif call_status == "in-progress" and direction != "inbound":
            # ⚠️ IGNORE - This is Twilio's media-active notification
            # We use first media packet detection instead (VAPI-style)
            logger.info(
                "ℹ️ IN-PROGRESS STATUS RECEIVED (ignored - using first media packet instead)"
            )
            logger.debug("🔍 DEBUG: Media stream status from Twilio (not user pickup)")
            logger.debug(
                "🔍 DEBUG: User pickup detection happens in bidirectional_stream.py"
            )

            # Don't do anything - first media packet will handle it

            return HTMLResponse("", media_type="application/xml")
        elif call_status == "in-progress" and direction == "inbound":
            logger.info("📞 INBOUND CALL IN-PROGRESS - SID: %s", call_sid)
            return HTMLResponse("", media_type="application/xml")

        elif call_status == "completed":
            # Call completed
            logger.info("📞 CALL COMPLETED - SID: %s", call_sid)

            # Fire call.completed webhook
            if call_session:
                try:
                    from app.services.webhook_service import fire_webhooks

                    background_tasks.add_task(
                        fire_webhooks,
                        call_session.tenant_id,
                        "call.completed",
                        {
                            "call_session_id": str(call_session.id),
                            "call_sid": call_sid,
                            "direction": direction,
                            "from_number": from_number,
                            "to_number": to_number,
                            "duration": call_session.duration,
                        },
                    )
                except Exception as _wh_exc:
                    logger.warning("call.completed webhook fire failed: %s", _wh_exc)

            return HTMLResponse("", media_type="application/xml")

        elif call_status == "failed":
            # Call failed - handle error
            logger.error("Call failed - SID: %s", call_sid)

            # Persist the "failed" status set earlier in this function (Blocker B
            # follow-up: previously never committed for this branch).
            _commit_terminal_call_session_status(db, call_session)

            # Fire call.failed webhook
            if call_session:
                try:
                    from app.services.webhook_service import fire_webhooks

                    background_tasks.add_task(
                        fire_webhooks,
                        call_session.tenant_id,
                        "call.failed",
                        {
                            "call_session_id": str(call_session.id),
                            "call_sid": call_sid,
                            "direction": direction,
                            "from_number": from_number,
                            "to_number": to_number,
                            "reason": "failed",
                        },
                    )
                except Exception as _wh_exc:
                    logger.warning("call.failed webhook fire failed: %s", _wh_exc)

            # Broadcast call failed event (non-blocking - fire and forget)
            if call_session:
                try:
                    asyncio.create_task(
                        broadcast_call_status_update(
                            call_session_id=str(call_session.id),
                            status="failed",
                            metadata={
                                "call_sid": call_sid,
                                "direction": direction,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                    )
                    logger.debug(
                        "✅ Queued call failed event for session %s", call_session.id
                    )

                    # Also broadcast call ended event for failed calls (non-blocking - fire and forget)
                    asyncio.create_task(
                        broadcast_call_ended(
                            call_session_id=str(call_session.id),
                            reason="failed",
                            final_data={
                                "call_sid": call_sid,
                                "direction": direction,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "duration": 0,
                            },
                        )
                    )
                    logger.debug(
                        "✅ Queued call ended (failed) event for session %s", call_session.id
                    )
                except Exception as e:
                    logger.error("❌ Failed to broadcast call failed event: %s", e)

                # Stop credit monitoring when call fails
                try:
                    credit_service.stop_credit_monitoring(call_session.id)
                    logger.debug(
                        "✅ Stopped credit monitoring for failed call session %s", call_session.id
                    )
                except Exception as e:
                    logger.warning(
                        "⚠️ Failed to stop credit monitoring (non-critical): %s", e
                    )

                try:
                    await notify_batch_call_ended(db, call_session.id, call_status)
                except Exception as batch_exc:
                    logger.warning(
                        "Batch call completion hook failed: %s",
                        batch_exc,
                        exc_info=True,
                    )

            return HTMLResponse("", media_type="application/xml")

        elif call_status == "canceled":
            # Call canceled via REST API before it was answered — same "never
            # completed" business effects as "failed" (GAP 4 follow-up): fire the
            # webhook, stop credit monitoring, and notify batch-call completion so
            # a canceled outbound batch call doesn't get stuck as permanently "active".
            logger.info("Call canceled - SID: %s", call_sid)

            # Persist the "failed" status set earlier in this function (Blocker B
            # follow-up: previously never committed for this branch).
            _commit_terminal_call_session_status(db, call_session)

            # Fire call.failed webhook (no dedicated call.canceled event type)
            if call_session:
                try:
                    from app.services.webhook_service import fire_webhooks

                    background_tasks.add_task(
                        fire_webhooks,
                        call_session.tenant_id,
                        "call.failed",
                        {
                            "call_session_id": str(call_session.id),
                            "call_sid": call_sid,
                            "direction": direction,
                            "from_number": from_number,
                            "to_number": to_number,
                            "reason": "canceled",
                        },
                    )
                except Exception as _wh_exc:
                    logger.warning(
                        "call.failed (canceled) webhook fire failed: %s", _wh_exc
                    )

            # Broadcast call canceled event (non-blocking - fire and forget)
            if call_session:
                try:
                    asyncio.create_task(
                        broadcast_call_status_update(
                            call_session_id=str(call_session.id),
                            status="failed",
                            metadata={
                                "call_sid": call_sid,
                                "twilio_status": call_status,
                                "direction": direction,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                    )
                    logger.debug(
                        "✅ Queued call canceled event for session %s", call_session.id
                    )

                    asyncio.create_task(
                        broadcast_call_ended(
                            call_session_id=str(call_session.id),
                            reason="canceled",
                            final_data={
                                "call_sid": call_sid,
                                "direction": direction,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "duration": 0,
                            },
                        )
                    )
                    logger.debug(
                        "✅ Queued call ended (canceled) event for session %s", call_session.id
                    )
                except Exception as e:
                    logger.error("❌ Failed to broadcast call canceled event: %s", e)

                # Stop credit monitoring when call is canceled
                try:
                    credit_service.stop_credit_monitoring(call_session.id)
                    logger.debug(
                        "✅ Stopped credit monitoring for canceled call session %s", call_session.id
                    )
                except Exception as e:
                    logger.warning(
                        "⚠️ Failed to stop credit monitoring (non-critical): %s", e
                    )

                try:
                    await notify_batch_call_ended(db, call_session.id, call_status)
                except Exception as batch_exc:
                    logger.warning(
                        "Batch call completion hook failed: %s",
                        batch_exc,
                        exc_info=True,
                    )

            return HTMLResponse("", media_type="application/xml")

        elif call_status in ("busy", "no-answer"):
            # Both busy and no-answer → internal "no_answer" (per ticket spec)
            logger.info("Call %s (internal: no_answer) - SID: %s", call_status, call_sid)

            # Persist the "no_answer" status set earlier in this function (Blocker B
            # follow-up: previously never committed for this branch).
            _commit_terminal_call_session_status(db, call_session)

            # Fire call.failed webhook for no_answer/busy
            if call_session:
                try:
                    from app.services.webhook_service import fire_webhooks

                    background_tasks.add_task(
                        fire_webhooks,
                        call_session.tenant_id,
                        "call.failed",
                        {
                            "call_session_id": str(call_session.id),
                            "call_sid": call_sid,
                            "direction": direction,
                            "from_number": from_number,
                            "to_number": to_number,
                            "reason": "no_answer",
                            "twilio_status": call_status,
                        },
                    )
                except Exception as _wh_exc:
                    logger.warning(
                        "call.failed (no_answer) webhook fire failed: %s", _wh_exc
                    )

            if call_session:
                try:
                    asyncio.create_task(
                        broadcast_call_status_update(
                            call_session_id=str(call_session.id),
                            status="no_answer",
                            metadata={
                                "call_sid": call_sid,
                                "twilio_status": call_status,
                                "direction": direction,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                    )
                    asyncio.create_task(
                        broadcast_call_ended(
                            call_session_id=str(call_session.id),
                            reason="no_answer",
                            final_data={
                                "call_sid": call_sid,
                                "twilio_status": call_status,
                                "direction": direction,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "duration": 0,
                            },
                        )
                    )
                except Exception as e:
                    logger.error("❌ Failed to broadcast no_answer event: %s", e)
                try:
                    credit_service.stop_credit_monitoring(call_session.id)
                except Exception as e:
                    logger.warning(
                        "⚠️ Failed to stop credit monitoring (non-critical): %s", e
                    )

                try:
                    await notify_batch_call_ended(db, call_session.id, call_status)
                except Exception as batch_exc:
                    logger.warning(
                        "Batch call completion hook failed: %s",
                        batch_exc,
                        exc_info=True,
                    )
            return HTMLResponse("", media_type="application/xml")

        else:
            # Default response for other statuses
            logger.info(
                "Unhandled call status: '%s' - using default response", call_status
            )
            response = VoiceResponse()
            text = "Thanks for calling! Have a great day!"
            lang = agent.language if agent and agent.language else "en"
            voice = agent.voice_type if agent and agent.voice_type else "female"
            tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(text)}&lang={lang}&voice={voice}"
            response.play(tts_url)
            return HTMLResponse(str(response), media_type="application/xml")

    except Exception as e:
        logger.error("ERROR occurred: %s", str(e), exc_info=True)
        logger.error("=== Call Events Webhook Failed ===")
        raise


@router.get("/dashboard/analytics", response_model=SuccessResponse[dict])
async def get_dashboard_analytics(
    agent_id: str | None = Query(None, description="Filter by specific agent ID"),
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """
    Thin wrapper that delegates to `voice_analytics_service`.
    """
    try:
        tenant_id = user.current_tenant_id

        if agent_id:
            try:
                uuid.UUID(agent_id)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid agent ID format")

        analytics_data = voice_analytics_service.get_dashboard_analytics(
            db=db,
            tenant_id=tenant_id,
            agent_id=agent_id,
        )

        message = f"Retrieved dashboard analytics for tenant {tenant_id}"
        if agent_id:
            message += f" filtered by agent {agent_id}"

        return create_success_response(analytics_data, message)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get dashboard analytics: {str(e)}",
        )


# Twilio's documented recording resource path (GAP 2 / SSRF hardening):
# https://www.twilio.com/docs/voice/api/recording#recording-uris
# https://api.twilio.com/2010-04-01/Accounts/{AccountSid}/Recordings/{RecordingSid}
_TWILIO_RECORDING_URL_PATH_RE = re.compile(
    r"^/2010-04-01/Accounts/[A-Za-z0-9]+/Recordings/[A-Za-z0-9]+$"
)


def _build_authenticated_recording_url(
    recording_url: str, account_sid: str, auth_token: str
) -> str | None:
    """
    Validate that `recording_url` is a genuine Twilio recording resource before
    injecting credentials and fetching it, and build the authenticated URL.

    Returns None if `recording_url` doesn't match Twilio's documented recording
    URL shape — callers must reject the request rather than fetching an
    attacker-controlled URL (SSRF guard).
    """
    if recording_url.startswith("http://") or recording_url.startswith("https://"):
        parsed = urlparse(recording_url)
        if parsed.scheme != "https" or parsed.netloc != "api.twilio.com":
            return None
        path = parsed.path
    else:
        # Twilio also sends relative recording URLs on some webhook shapes.
        path = recording_url

    if not _TWILIO_RECORDING_URL_PATH_RE.match(path):
        return None

    return f"https://{account_sid}:{auth_token}@api.twilio.com{path}.wav"


@router.post("/webhook/recording-callback", response_class=HTMLResponse)
async def handle_recording_callback(
    request: Request,
    agentId: str | None = Query(None),
    userId: str | None = Query(None),
    callSessionId: str | None = Query(None),
    body: str = Depends(get_request_body),
    db: Session = Depends(get_db),
):
    """
    VAPI-style Recording Callback Webhook

    When user stops speaking (silence detected), Twilio sends the recording here.
    We download it, transcribe with Deepgram STT, generate LLM response, and return TwiML.

    This is the simple, synchronous approach similar to feature/openai branch.
    """
    logger.info("🎙️ RECORDING CALLBACK WEBHOOK - VAPI-style")
    logger.debug("📞 Call Session: %s", callSessionId)
    logger.debug("🤖 Agent: %s", agentId)

    try:
        form_data = await request.form()

        # Extract recording details
        recording_url = form_data.get("RecordingUrl", "")
        recording_sid = form_data.get("RecordingSid", "")
        recording_duration = form_data.get("RecordingDuration", "0")
        call_sid = form_data.get("CallSid", "")
        recording_status = form_data.get("RecordingStatus", "")

        # Resolve call session up front so per-tenant Twilio credentials are
        # available for signature validation below.
        call_session = None
        agent = None
        if callSessionId:
            try:
                session_uuid = uuid.UUID(callSessionId)
                call_session = call_session_service.get_call_session_by_id(
                    db, session_uuid
                )
                if call_session and agentId:
                    agent = agent_service.get_agent_by_id(
                        db, uuid.UUID(agentId), call_session.tenant_id
                    )
                    logger.debug(
                        "✅ Found call session and agent: %s", agent.name if agent else 'Unknown'
                    )
            except ValueError:
                logger.warning("⚠️ Invalid call session ID: %s", callSessionId)

        form_params = dict(form_data)
        if not await _validate_transfer_webhook_signature(
            request, db, call_session, form_params
        ):
            logger.warning(
                "Recording callback webhook: invalid Twilio signature (call_sid=%s, callSessionId=%s)",
                call_sid,
                callSessionId,
            )
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")

        logger.debug("🎵 Recording URL: %s", recording_url)
        logger.debug("📝 Recording SID: %s", recording_sid)
        logger.debug("⏱️ Duration: %ss", recording_duration)
        logger.debug("📊 Status: %s", recording_status)

        # IMPORTANT: Twilio calls this webhook twice:
        # 1. 'action' callback (no status, has URL) - User finished speaking → PROCESS THIS for TTS
        # 2. 'recordingStatusCallback' (has status) - Recording processed → SKIP (just for logging)

        if recording_status:
            # This is a status callback, not the action callback
            # We don't need to return TTS here, just acknowledge
            logger.debug(
                "ℹ️ Recording status callback (status=%s) - acknowledging only, no TTS", recording_status
            )
            return HTMLResponse("", media_type="application/xml")

        # If no recording URL at all, something is wrong
        if not recording_url:
            logger.warning("⚠️ No recording URL provided - cannot process")
            return HTMLResponse("", media_type="application/xml")

        # This is the 'action' callback - user finished speaking
        # Process this for TTS response
        logger.info("✅ Action callback detected - processing for TTS response")

        # Process recording if available
        if recording_url and call_session:
            try:
                import requests

                # ✅ Get Twilio credentials based on call session (DB or Env)
                account_sid, auth_token = get_twilio_credentials_for_call(
                    db, call_session
                )

                # Build authenticated recording URL — strictly validated against
                # Twilio's documented recording resource shape (SSRF guard, GAP 2).
                auth_url = _build_authenticated_recording_url(
                    recording_url, account_sid, auth_token
                )
                if auth_url is None:
                    logger.error(
                        "❌ Rejected recording URL that does not match Twilio's recording resource pattern: %s",
                        recording_url,
                    )
                    raise Exception("Invalid Twilio recording URL")

                logger.debug("📥 Downloading audio from Twilio...")

                # Download the recording
                audio_response = requests.get(auth_url, timeout=10)

                if audio_response.status_code != 200:
                    logger.error(
                        "❌ Failed to download recording: HTTP %s", audio_response.status_code
                    )
                    raise Exception(
                        f"Failed to download recording: HTTP {audio_response.status_code}"
                    )

                audio_content = audio_response.content
                logger.debug("✅ Downloaded %s bytes of audio", len(audio_content))

                language_code = (settings.DEEPGRAM_STT_LANGUAGE or "en").strip()

                logger.debug(
                    "🎙️ Transcribing with Deepgram STT (language: %s)...", language_code
                )

                from app.services.deepgram_stt_service import deepgram_stt_service

                stt_result = await deepgram_stt_service.transcribe_audio_chunk(
                    audio_content=audio_content, language_code=language_code
                )

                transcript = stt_result.get("transcript", "").strip()
                confidence = stt_result.get("confidence", 0.0)

                logger.info("📝 Deepgram STT Transcript: '%s'", transcript)
                logger.debug("📊 Confidence: %s", format(confidence, '.2f'))

                # If we have a transcript, process it
                if transcript:
                    # Add user speech to transcript
                    await add_to_transcript(
                        call_session,
                        "client",
                        transcript,
                        db,
                        message_type="speech",
                        confidence=confidence,
                    )

                    # Log voice interaction
                    await VoiceLoggingService.log_voice_interaction(
                        db=db,
                        call_session_id=call_session.id,
                        interaction_type="speech_input",
                        speech_text=transcript,
                        confidence=confidence,
                        duration=(
                            float(recording_duration) if recording_duration else None
                        ),
                        metadata={
                            "call_sid": call_sid,
                            "recording_sid": recording_sid,
                            "agent_id": str(agent.id) if agent else None,
                            "source": "deepgram_stt",
                        },
                    )

                    # Generate agent response using LLM
                    logger.debug("🤖 Generating agent response...")
                    response_text = await VoiceLoggingService.generate_agent_response(
                        speech_text=transcript,
                        confidence=confidence,
                        agent=agent,
                        db=db,
                        call_session_id=call_session.id,
                    )

                    logger.info("✅ Agent response: '%s'", response_text)

                    # Add agent response to transcript
                    await add_to_transcript(
                        call_session,
                        "agent",
                        response_text,
                        db,
                        message_type="agent_response",
                    )

                    # Check if this is a goodbye
                    is_goodbye = VoiceLoggingService._is_completion_goodbye(
                        response_text
                    )
                    if is_goodbye:
                        logger.info("🛑 Goodbye detected - ending call")
                        response = VoiceResponse()
                        response.hangup()
                        twiml_str = str(response)
                        logger.debug(
                            "📤 Returning TwiML (goodbye): %s...", twiml_str[:200]
                        )
                        return HTMLResponse(twiml_str, media_type="application/xml")

                    # Store TTS text in call session metadata for WebSocket to retrieve
                    lang = agent.language if agent and agent.language else "en"
                    voice_type = (
                        agent.voice_type if agent and agent.voice_type else "female"
                    )

                    if not call_session.call_metadata:
                        call_session.call_metadata = {}

                    call_session.call_metadata["pending_tts"] = {
                        "text": response_text,
                        "lang": lang,
                        "voice": voice_type,
                    }
                    db.commit()

                    logger.debug(
                        "💾 Stored pending TTS in metadata: '%s...'", response_text[:50]
                    )

                    # Build TwiML for TTS-only WebSocket streaming + Recording
                    recording_callback_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/recording-callback?agentId={agentId}&userId={userId}&callSessionId={callSessionId}"

                    from app.routers.bidirectional_stream import build_tts_only_twiml

                    twiml_str = build_tts_only_twiml(
                        call_session_id=str(call_session.id),
                        agent_id=str(agent.id) if agent else agentId,
                        record_callback_url=recording_callback_url,
                    )

                    logger.debug("🎵 Returning TwiML with TTS WebSocket streaming")
                    logger.debug("📤 TwiML: %s...", twiml_str[:200])
                    return HTMLResponse(twiml_str, media_type="application/xml")

                else:
                    # No transcript - ask user to repeat
                    logger.info("⚠️ No transcript from Deepgram STT")
                    response = VoiceResponse()

                    # Natural "didn't catch that" response
                    text = get_random_didnt_catch_response()
                    lang = agent.language if agent and agent.language else "en"
                    voice = agent.voice_type if agent and agent.voice_type else "female"
                    tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(text)}&lang={lang}&voice={voice}"
                    response.play(tts_url)

                    # Record again
                    recording_callback_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/recording-callback?agentId={agentId}&userId={userId}&callSessionId={callSessionId}"

                    response.record(
                        action=recording_callback_url,
                        method="POST",
                        timeout=3,  # Faster detection
                        max_length=60,
                        play_beep=False,
                        trim="do-not-trim",
                        recording_status_callback=recording_callback_url,
                        recording_status_callback_method="POST",
                        transcribe=False,
                    )

                    return HTMLResponse(str(response), media_type="application/xml")

            except Exception as e:
                logger.error("❌ Error processing recording: %s", e, exc_info=True)

                # Fallback response
                response = VoiceResponse()
                text = "Sorry, I had trouble hearing you. Could you please repeat that?"
                lang = agent.language if agent and agent.language else "en"
                voice = agent.voice_type if agent and agent.voice_type else "female"
                tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(text)}&lang={lang}&voice={voice}"
                response.play(tts_url)

                recording_callback_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/recording-callback?agentId={agentId}&userId={userId}&callSessionId={callSessionId}"

                response.record(
                    action=recording_callback_url,
                    method="POST",
                    timeout=3,  # Faster detection
                    max_length=60,
                    play_beep=False,
                    trim="do-not-trim",
                    recording_status_callback=recording_callback_url,
                    recording_status_callback_method="POST",
                    transcribe=False,
                )

                return HTMLResponse(str(response), media_type="application/xml")

        # Fallback if no recording URL
        logger.warning("⚠️ No recording URL provided")
        response = VoiceResponse()
        text = "I didn't hear anything. Please try speaking again."
        lang = agent.language if agent and agent.language else "en"
        voice = agent.voice_type if agent and agent.voice_type else "female"
        tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(text)}&lang={lang}&voice={voice}"
        response.play(tts_url)

        recording_callback_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/recording-callback?agentId={agentId}&userId={userId}&callSessionId={callSessionId}"

        response.record(
            action=recording_callback_url,
            method="POST",
            timeout=3,  # Faster detection
            max_length=60,
            play_beep=False,
            trim="do-not-trim",
            recording_status_callback=recording_callback_url,
            recording_status_callback_method="POST",
            transcribe=False,
        )

        return HTMLResponse(str(response), media_type="application/xml")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ Error in recording callback webhook: %s", e, exc_info=True)

        # Ultimate fallback - use streaming TwiML if we have session info
        if call_session and agent:
            streaming_twiml = build_streaming_twiml(str(call_session.id), str(agent.id))
            return HTMLResponse(streaming_twiml, media_type="application/xml")
        else:
            # Fallback to simple response if no session info
            response = VoiceResponse()
            response.say(
                "Sorry, something went wrong. Please try calling again later. Goodbye!"
            )
            response.hangup()
            return HTMLResponse(str(response), media_type="application/xml")


@router.post("/webhook/gather-speech", response_class=HTMLResponse)
async def handle_gather_speech_webhook(
    request: Request,
    agentId: str | None = Query(None),
    callSessionId: str | None = Query(None),
    body: str = Depends(get_request_body),
    db: Session = Depends(get_db),
):
    """
    DEPRECATED: This endpoint was used for the old Gather-based approach.
    Now we use the simpler /webhook/recording-callback endpoint with <Record>.

    Keeping this for backward compatibility with feature/openai branch style.
    """
    logger.warning("⚠️ DEPRECATED: GATHER SPEECH WEBHOOK CALLED")
    logger.warning("Use /webhook/recording-callback instead")

    try:
        form_data = await request.form()

        call_sid = form_data.get("CallSid", "")
        recording_url = form_data.get("RecordingUrl", "")
        speech_result = form_data.get("SpeechResult", "")  # Twilio's transcription
        confidence = form_data.get("Confidence", "0")

        logger.debug("📞 Call SID: %s", call_sid)
        logger.debug("🎤 Twilio Speech Result: %s", speech_result)
        logger.debug("📊 Confidence: %s", confidence)
        logger.debug("🎵 Recording URL: %s", recording_url)

        # Get call session
        call_session = None
        if callSessionId:
            try:
                session_uuid = uuid.UUID(callSessionId)
                call_session = call_session_service.get_call_session_by_id(
                    db, session_uuid
                )
                logger.debug("✅ Found call session: %s", call_session.id)
            except ValueError:
                logger.warning("⚠️ Invalid call session ID: %s", callSessionId)

        # Get agent
        agent = None
        if agentId and call_session:
            try:
                agent = agent_service.get_agent_by_id(
                    db, uuid.UUID(agentId), call_session.tenant_id
                )
                logger.debug("✅ Agent: %s", agent.name)
            except Exception as e:
                logger.warning("⚠️ Error fetching agent: %s", e)

        form_params = dict(form_data)
        if not await _validate_transfer_webhook_signature(
            request, db, call_session, form_params
        ):
            logger.warning(
                "Gather speech webhook: invalid Twilio signature (call_sid=%s, callSessionId=%s)",
                call_sid,
                callSessionId,
            )
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")

        # Download audio from Twilio recording
        if recording_url and call_session:
            try:
                import requests

                # Get Twilio credentials
                client = twilio_service.get_client()
                account_sid = client.username
                auth_token = client.password

                # Download recording with authentication — strictly validated
                # against Twilio's documented recording resource shape (SSRF guard).
                auth_url = _build_authenticated_recording_url(
                    recording_url, account_sid, auth_token
                )
                if auth_url is None:
                    logger.error(
                        "❌ Rejected recording URL that does not match Twilio's recording resource pattern: %s",
                        recording_url,
                    )
                    raise Exception("Invalid Twilio recording URL")
                logger.debug("📥 Downloading audio from Twilio...")

                audio_response = requests.get(auth_url, timeout=10)
                audio_content = audio_response.content

                logger.debug("✅ Downloaded %s bytes of audio", len(audio_content))

                from app.services.deepgram_stt_service import deepgram_stt_service

                language_code = (settings.DEEPGRAM_STT_LANGUAGE or "en").strip()

                logger.debug(
                    "🎙️ Transcribing with Deepgram STT (language: %s)...", language_code
                )

                stt_result = await deepgram_stt_service.transcribe_audio_chunk(
                    audio_content=audio_content, language_code=language_code
                )

                dg_transcript = stt_result.get("transcript", "")
                dg_confidence = stt_result.get("confidence", 0.0)

                logger.info("📝 Deepgram STT Transcript: '%s'", dg_transcript)
                logger.debug("📊 Deepgram STT Confidence: %s", format(dg_confidence, '.2f'))

                # Use Deepgram transcript (more accurate)
                final_transcript = dg_transcript if dg_transcript else speech_result

                if final_transcript:
                    # Add to transcript
                    await add_to_transcript(
                        call_session,
                        "client",
                        final_transcript,
                        db,
                        message_type="speech",
                        confidence=dg_confidence,
                    )

                    # Generate LLM response
                    response_text = await VoiceLoggingService.generate_agent_response(
                        speech_text=final_transcript,
                        confidence=dg_confidence,
                        agent=agent,
                        db=db,
                        call_session_id=call_session.id,
                    )

                    # Add agent response to transcript
                    await add_to_transcript(
                        call_session,
                        "agent",
                        response_text,
                        db,
                        message_type="agent_response",
                    )

                    logger.info("✅ Generated agent response: '%s'", response_text)

                    # Create response TwiML
                    response = VoiceResponse()

                    # Say agent response using Google TTS
                    lang = agent.language if agent and agent.language else "en"
                    voice = agent.voice_type if agent and agent.voice_type else "female"
                    tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(response_text)}&lang={lang}&voice={voice}"
                    response.play(tts_url)

                    # Check if goodbye
                    is_goodbye = VoiceLoggingService._is_completion_goodbye(
                        response_text
                    )
                    if is_goodbye:
                        response.hangup()
                        logger.info("🛑 Goodbye detected - ending call")
                        return HTMLResponse(str(response), media_type="application/xml")

                    # Continue conversation - gather next input
                    response.gather(
                        input="speech",
                        timeout=10,
                        speech_timeout="auto",
                        action=f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/gather-speech?agentId={agentId}&callSessionId={call_session.id}",
                        method="POST",
                        enhanced=True,
                        profanity_filter=False,
                        language="en-US",
                    )

                    # Fallback
                    text = "I didn't catch that. Please try again!"
                    lang = agent.language if agent and agent.language else "en"
                    voice = agent.voice_type if agent and agent.voice_type else "female"
                    tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(text)}&lang={lang}&voice={voice}"
                    response.play(tts_url)
                    response.redirect(
                        f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events?agentId={agentId}&callSessionId={call_session.id}",
                        method="POST",
                    )

                    logger.debug("📝 Response TwiML: %s...", str(response)[:200])
                    return HTMLResponse(str(response), media_type="application/xml")

            except Exception as e:
                logger.error("❌ Error processing gathered speech: %s", e, exc_info=True)

        # Fallback response
        response = VoiceResponse()
        text = "I didn't hear you. Could you please repeat that?"
        lang = agent.language if agent and agent.language else "en"
        voice = agent.voice_type if agent and agent.voice_type else "female"
        tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(text)}&lang={lang}&voice={voice}"
        response.play(tts_url)

        response.gather(
            input="speech",
            timeout=10,
            speech_timeout="auto",
            action=f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/gather-speech?agentId={agentId}&callSessionId={call_session.id}",
            method="POST",
            enhanced=True,
            profanity_filter=False,
            language="en-US",
        )

        return HTMLResponse(str(response), media_type="application/xml")

    except Exception as e:
        logger.error("❌ Error in gather speech webhook: %s", e, exc_info=True)
        raise


@router.post("/webhook/recording-status")
async def handle_recording_status_webhook(
    request: Request, db: Session = Depends(get_db)
):
    """
    Handle Twilio recording status callbacks.
    This webhook is called when recording status changes (in-progress, completed, etc.)
    """
    try:
        form_data = await request.form()

        # Extract recording information
        recording_sid = form_data.get("RecordingSid")
        call_sid = form_data.get("CallSid")
        recording_status = form_data.get("RecordingStatus")
        recording_url = form_data.get("RecordingUrl")
        recording_duration = form_data.get("RecordingDuration")

        logger.info("🎙️ RECORDING STATUS UPDATE")
        logger.debug("Recording SID: %s", recording_sid)
        logger.debug("Call SID: %s", call_sid)
        logger.debug("Status: %s", recording_status)
        logger.debug("URL: %s", recording_url)
        logger.debug("Duration: %s", recording_duration)

        # Find the call session
        call_session = (
            call_session_service.get_call_session_by_twilio_sid(db, call_sid)
            if call_sid
            else None
        )

        form_params = dict(form_data)
        if not await _validate_transfer_webhook_signature(
            request, db, call_session, form_params
        ):
            logger.warning(
                "Recording status webhook: invalid Twilio signature (call_sid=%s)",
                call_sid,
            )
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")

        if call_sid:
            if call_session:
                # Update recording URL when recording is completed
                if recording_status == "completed" and recording_url:
                    call_session.recording_url = recording_url
                    db.commit()
                    logger.info(
                        "✅ Updated call session %s with recording URL", call_session.id
                    )

                    # Broadcast call status update when recording is completed (non-blocking - fire and forget)
                    try:
                        asyncio.create_task(
                            broadcast_call_status_update(
                                call_session_id=str(call_session.id),
                                status="completed",
                                metadata={
                                    "call_sid": call_sid,
                                    "call_duration": recording_duration,
                                    "message": "Call completed",
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                },
                            )
                        )
                        logger.debug(
                            "✅ Queued recording completed status update for session %s", call_session.id
                        )
                    except Exception as e:
                        logger.warning(
                            "⚠️ Failed to queue recording completed status update (non-critical): %s", e
                        )
                else:
                    logger.debug(
                        "📝 Recording status: %s - URL not ready yet", recording_status
                    )
            else:
                logger.warning("⚠️ Call session not found for SID: %s", call_sid)

        # Return empty TwiML response
        return HTMLResponse("", media_type="application/xml")

    except HTTPException:
        raise
    except Exception as e:
        logger.warning("⚠️ Error handling recording status webhook: %s", e)
        return HTMLResponse("", media_type="application/xml")


async def _validate_transfer_webhook_signature(
    request: Request,
    db: Session,
    call_session: CallSession,
    form_params: dict,
) -> bool:
    if settings.ALLOW_UNAUTHENTICATED_WEBHOOKS:
        return True
    try:
        _, auth_token = get_twilio_credentials_for_call(db, call_session)
        return validate_twilio_signature_with_token(request, form_params, auth_token)
    except Exception as cred_err:
        logger.warning(
            "Transfer webhook: per-session Twilio token unavailable (%s); falling back to env token",
            cred_err,
        )
        return validate_twilio_signature(request, form_params)


def _conference_room_name(session_id: uuid.UUID) -> str:
    return f"warm-tr-{session_id}"


@router.post(
    "/webhook/transfer/dial-cold",
    response_class=HTMLResponse,
    include_in_schema=False,
)
async def transfer_webhook_dial_cold(
    request: Request,
    callSessionId: str = Query(...),
    db: Session = Depends(get_db),
):
    """Twilio fetches this after redirect; cold transfer dials the configured route number."""
    try:
        session_uuid = uuid.UUID(callSessionId)
    except ValueError:
        vr = VoiceResponse()
        vr.say("Sorry, this transfer link is invalid.")
        vr.hangup()
        return HTMLResponse(str(vr), media_type="application/xml")

    call_session = call_session_service.get_call_session_by_id(db, session_uuid)
    if not call_session:
        vr = VoiceResponse()
        vr.say("Sorry, we could not complete the transfer.")
        vr.hangup()
        return HTMLResponse(str(vr), media_type="application/xml")

    form_data = await request.form()
    form_params = dict(form_data)
    if not await _validate_transfer_webhook_signature(
        request, db, call_session, form_params
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid Twilio signature"
        )

    twilio_sid = str(
        form_params.get("CallSid") or request.query_params.get("CallSid") or ""
    )
    if (
        call_session.twilio_call_sid
        and twilio_sid
        and twilio_sid != call_session.twilio_call_sid
    ):
        logger.warning(
            "Transfer dial-cold CallSid mismatch session=%s expected=%s got=%s",
            call_session.id,
            call_session.twilio_call_sid,
            twilio_sid,
        )
        vr = VoiceResponse()
        vr.say("Sorry, we could not verify this call.")
        vr.hangup()
        return HTMLResponse(str(vr), media_type="application/xml")

    agent = (
        db.query(Agent)
        .options(joinedload(Agent.transfer_route))
        .filter(
            Agent.id == call_session.agent_id,
            Agent.tenant_id == call_session.tenant_id,
        )
        .first()
    )
    route = getattr(agent, "transfer_route", None) if agent else None
    if (
        not route
        or getattr(route, "is_deleted", False)
        or (route.transfer_type or "").lower() != "cold"
    ):
        vr = VoiceResponse()
        vr.say("Sorry, a human transfer is not available right now.")
        vr.hangup()
        return HTMLResponse(str(vr), media_type="application/xml")

    caller_id = twilio_caller_id_for_transfer_dial(call_session)
    if not caller_id:
        logger.warning(
            "Transfer dial-cold: no Twilio caller ID for session %s (type=%s)",
            call_session.id,
            call_session.call_type,
        )
        vr = VoiceResponse()
        vr.say("Sorry, transfer is not available on this line configuration.")
        vr.hangup()
        return HTMLResponse(str(vr), media_type="application/xml")

    sid_str = str(call_session.id)
    action_url = (
        f"{settings.WEBHOOK_BASE_URL.rstrip('/')}/api/v1/voice/webhook/transfer/dial-complete"
        f"?callSessionId={sid_str}"
    )
    vr = VoiceResponse()
    vr.say("Connecting you now.")
    dial = vr.dial(caller_id=caller_id, timeout=45, action=action_url, method="POST")
    dial.number(route.phone_number)
    return HTMLResponse(str(vr), media_type="application/xml")


@router.post(
    "/webhook/transfer/dial-complete",
    response_class=HTMLResponse,
    include_in_schema=False,
)
async def transfer_webhook_dial_complete(
    request: Request,
    callSessionId: str = Query(...),
    db: Session = Depends(get_db),
):
    """After cold Dial ends (busy/no-answer), hang up gracefully. Twilio-signed."""

    def _hangup_twiml(message: str | None = None) -> HTMLResponse:
        vr = VoiceResponse()
        if message:
            vr.say(message)
        vr.hangup()
        return HTMLResponse(str(vr), media_type="application/xml")

    try:
        session_uuid = uuid.UUID(callSessionId)
    except ValueError:
        return _hangup_twiml("Sorry, this request is invalid.")

    call_session = call_session_service.get_call_session_by_id(db, session_uuid)
    if not call_session:
        return _hangup_twiml()

    form_data = await request.form()
    form_params = dict(form_data)
    if not await _validate_transfer_webhook_signature(
        request, db, call_session, form_params
    ):
        logger.warning(
            "dial-complete: invalid Twilio signature session=%s", callSessionId
        )
        return _hangup_twiml()

    parent_sid = str(form_params.get("CallSid") or "")
    if (
        call_session.twilio_call_sid
        and parent_sid
        and parent_sid != call_session.twilio_call_sid
    ):
        logger.warning(
            "dial-complete CallSid mismatch session=%s expected=%s got=%s",
            call_session.id,
            call_session.twilio_call_sid,
            parent_sid,
        )
        return _hangup_twiml()

    dial_status = str(form_params.get("DialCallStatus") or "")
    vr = VoiceResponse()
    if dial_status and dial_status not in ("completed",):
        vr.say("We could not reach someone right now. Goodbye.")
    vr.hangup()

    # Status Webhook — "transfer" event. This is the one spot where the cold
    # transfer's outcome (answered vs busy/no-answer/failed) is actually known;
    # not fired on the earlier dial-cold initiation webhook.
    try:
        _call_flow = (
            db.query(CallFlow)
            .filter(
                CallFlow.id == call_session.call_flow_id,
                CallFlow.tenant_id == call_session.tenant_id,
            )
            .first()
            if call_session.call_flow_id
            else None
        )
        if _call_flow and _call_flow.status_webhook_enabled:
            from app.services.system_webhook_service import schedule_status_webhook

            schedule_status_webhook(
                call_session.id,
                "call.transfer",
                extra={
                    "outcome": dial_status or "unknown",
                    "dial_call_sid": str(form_params.get("DialCallSid") or ""),
                },
            )
    except Exception as exc:
        logger.warning(
            "Status webhook (transfer) schedule failed (non-critical): %s", exc
        )

    return HTMLResponse(str(vr), media_type="application/xml")


@router.post(
    "/webhook/transfer/conference-customer",
    response_class=HTMLResponse,
    include_in_schema=False,
)
async def transfer_webhook_conference_customer(
    request: Request,
    callSessionId: str = Query(...),
    db: Session = Depends(get_db),
):
    """Put the customer leg into a Twilio conference (warm transfer step 1)."""
    try:
        session_uuid = uuid.UUID(callSessionId)
    except ValueError:
        vr = VoiceResponse()
        vr.say("Sorry, this transfer link is invalid.")
        vr.hangup()
        return HTMLResponse(str(vr), media_type="application/xml")

    call_session = call_session_service.get_call_session_by_id(db, session_uuid)
    if not call_session:
        vr = VoiceResponse()
        vr.say("Sorry, we could not complete the transfer.")
        vr.hangup()
        return HTMLResponse(str(vr), media_type="application/xml")

    form_data = await request.form()
    form_params = dict(form_data)
    if not await _validate_transfer_webhook_signature(
        request, db, call_session, form_params
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid Twilio signature"
        )

    twilio_sid = str(
        form_params.get("CallSid") or request.query_params.get("CallSid") or ""
    )
    if (
        call_session.twilio_call_sid
        and twilio_sid
        and twilio_sid != call_session.twilio_call_sid
    ):
        vr = VoiceResponse()
        vr.say("Sorry, we could not verify this call.")
        vr.hangup()
        return HTMLResponse(str(vr), media_type="application/xml")

    room = _conference_room_name(call_session.id)
    vr = VoiceResponse()
    vr.say("Please hold while we connect you to a team member.")
    dial = vr.dial()
    dial.conference(
        room,
        beep="false",
        start_conference_on_enter=True,
        end_conference_on_exit=False,
    )
    return HTMLResponse(str(vr), media_type="application/xml")


@router.post(
    "/webhook/transfer/conference-supervisor",
    response_class=HTMLResponse,
    include_in_schema=False,
)
async def transfer_webhook_conference_supervisor(
    request: Request,
    callSessionId: str = Query(...),
    db: Session = Depends(get_db),
):
    """Outbound supervisor leg joins the same conference as the customer."""
    try:
        session_uuid = uuid.UUID(callSessionId)
    except ValueError:
        vr = VoiceResponse()
        vr.say("Invalid transfer.")
        vr.hangup()
        return HTMLResponse(str(vr), media_type="application/xml")

    call_session = call_session_service.get_call_session_by_id(db, session_uuid)
    if not call_session:
        vr = VoiceResponse()
        vr.say("Invalid transfer.")
        vr.hangup()
        return HTMLResponse(str(vr), media_type="application/xml")

    form_data = await request.form()
    form_params = dict(form_data)
    if not await _validate_transfer_webhook_signature(
        request, db, call_session, form_params
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid Twilio signature"
        )

    room = _conference_room_name(call_session.id)
    vr = VoiceResponse()
    dial = vr.dial()
    dial.conference(
        room,
        beep="false",
        start_conference_on_enter=True,
        end_conference_on_exit=False,
    )
    return HTMLResponse(str(vr), media_type="application/xml")


@router.post("/call/end", response_model=SuccessResponse[dict])
async def end_call(
    request: dict, user: User = Depends(require_tenant), db: Session = Depends(get_db)
):
    """
    End a call programmatically

    Request Payload:
    {
        "callSessionId": "uuid",
        "reason": "user_requested" | "agent_completed" | "timeout" | "error",
        "message": "Optional goodbye message"
    }
    """
    try:
        call_session_id = request.get("callSessionId")
        reason = request.get("reason", "user_requested")
        goodbye_message = request.get(
            "message", "Thank you for calling! Have a great day!"
        )

        if not call_session_id:
            raise HTTPException(status_code=400, detail="callSessionId is required")

        # Get call session
        try:
            session_uuid = uuid.UUID(call_session_id)
            call_session = call_session_service.get_call_session_by_id(db, session_uuid)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid callSessionId format")

        if not call_session:
            raise HTTPException(status_code=404, detail="Call session not found")

        # Verify user has access to this call session
        if call_session.tenant_id != user.current_tenant_id:
            raise HTTPException(
                status_code=403, detail="Access denied to this call session"
            )

        # End the call using Twilio: same account that created the call (DB phone creds)
        # with env-based fallback (legacy / if DB mapping is missing)
        call_ended = False
        if call_session.twilio_call_sid:
            try:
                account_sid, auth_token = get_twilio_credentials_for_call(
                    db, call_session
                )
                call_ended = twilio_service.end_call_with_credentials(
                    call_session.twilio_call_sid, account_sid, auth_token
                )
            except Exception as cred_err:
                logger.warning(
                    "end_call: DB Twilio credentials unavailable for session %s (%s); "
                    "trying default env client",
                    call_session.id,
                    cred_err,
                )
                call_ended = twilio_service.end_call(call_session.twilio_call_sid)

        # Update call session status
        call_session.status = "completed"
        call_session.end_time = datetime.now(timezone.utc)

        if call_session.start_time:
            duration = (call_session.end_time - call_session.start_time).total_seconds()
            call_session.duration = int(duration)

        # Update call session AND call log together (single commit)
        call_session_service.update_call_session_status(
            db, call_session.id, "completed", ended_reason="completed"
        )
        try:
            maybe_update_resume_status_on_call_completed(db, call_session.id)
        except Exception as mq_exc:
            logger.warning(
                "Resume screening qualify on end_call: %s", mq_exc, exc_info=True
            )

        # Add goodbye message to transcript
        if goodbye_message:
            await add_to_transcript(
                call_session,
                "agent",
                goodbye_message,
                db,
                message_type="call_end",
                agent_id=call_session.agent_id,
                user_id=call_session.user_id,
            )

        # Broadcast call ended event
        try:
            asyncio.create_task(
                broadcast_call_ended(
                    call_session_id=str(call_session.id),
                    reason=reason,
                    final_data={
                        "call_sid": call_session.twilio_call_sid,
                        "duration": call_session.duration,
                        "end_time": call_session.end_time.isoformat(),
                        "transcript": call_session.call_transcript or [],
                    },
                )
            )
        except Exception as e:
            logger.warning("⚠️ Failed to broadcast call ended event: %s", e)

        return SuccessResponse(
            data={
                "callSessionId": str(call_session.id),
                "status": "completed",
                "reason": reason,
                "duration": call_session.duration,
                "twilioEnded": call_ended,
            },
            message="Call ended successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ Error ending call: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to end call")


@router.get("/recording/{call_session_id}/access")
async def get_recording_access(
    call_session_id: str,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """
    Stream call recording directly to user (NO Twilio login required!)
    Returns audio file that can be played directly in browser.
    """
    try:
        # Get call session and verify user has access
        call_session = (
            db.query(CallSession)
            .filter(
                CallSession.id == call_session_id,
                CallSession.tenant_id == user.current_tenant_id,
            )
            .first()
        )

        if not call_session:
            raise HTTPException(
                status_code=404, detail="Call session not found or access denied"
            )

        if not call_session.recording_url:
            raise HTTPException(
                status_code=404, detail="No recording available for this call"
            )

        # ✅ Get Twilio credentials based on call session (DB or Env)
        account_sid, auth_token = get_twilio_credentials_for_call(db, call_session)

        # Extract recording SID from the URL
        recording_sid = (
            call_session.recording_url.split("/")[-1]
            .replace(".mp3", "")
            .replace(".wav", "")
        )

        # Create authenticated Twilio URL for server-side download
        authenticated_url = f"https://{account_sid}:{auth_token}@api.twilio.com/2010-04-01/Accounts/{account_sid}/Recordings/{recording_sid}.mp3"

        logger.info("📥 Streaming recording for call session: %s", call_session_id)
        logger.debug("🎵 Recording SID: %s", recording_sid)

        # Download recording from Twilio (server-side with auth)
        response = requests.get(authenticated_url, stream=True, timeout=30)

        if response.status_code != 200:
            logger.error("❌ Failed to fetch recording: HTTP %s", response.status_code)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch recording from Twilio: HTTP {response.status_code}",
            )

        logger.info("✅ Streaming recording to user (no login required)")

        # Stream audio directly to user (NO authentication required on user's end!)
        return StreamingResponse(
            response.iter_content(chunk_size=8192),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"inline; filename=call_recording_{call_session_id}.mp3",
                "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
                "Accept-Ranges": "bytes",  # Enable seeking in audio player
            },
        )

    except HTTPException:
        raise
    except requests.RequestException as e:
        logger.error("❌ Network error fetching recording: %s", e)
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error("❌ Error streaming recording: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to stream recording: {str(e)}"
        )


@router.post(
    "/transcript/analyze/{call_session_id}", response_model=SuccessResponse[dict]
)
async def analyze_call_transcript(
    call_session_id: str,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """
    Analyze call transcript using LLM for summary, sentiment, and recommendations.
    """
    try:
        # Validate call session ID
        try:
            session_uuid = uuid.UUID(call_session_id)
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid call session ID format"
            )

        # Get call session
        call_session = call_session_service.get_call_session_by_id(db, session_uuid)
        if not call_session:
            raise HTTPException(status_code=404, detail="Call session not found")

        # Check if user has access to this call session
        if (
            call_session.user_id != user.id
            and call_session.tenant_id != user.current_tenant_id
        ):
            raise HTTPException(
                status_code=403, detail="Access denied to this call session"
            )

        analysis_result = voice_analysis_service.analyze_call_transcript(
            db=db,
            call_session=call_session,
            user_id=user.id,
        )

        try:
            from app.services.inbound_call_crm_sync_service import (
                schedule_inbound_crm_sync,
            )

            schedule_inbound_crm_sync(session_uuid)
        except Exception as crm_exc:
            logger.warning(
                "Inbound CRM refresh after transcript analysis skipped (non-critical): %s",
                crm_exc,
            )

        return create_success_response(
            data=analysis_result,
            message=f"Transcript analysis completed successfully using {analysis_result.get('model_used')}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ Error in transcript analysis endpoint: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
