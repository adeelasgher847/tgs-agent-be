import asyncio
from datetime import datetime, timezone
import uuid

from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from twilio.base.exceptions import TwilioRestException

from app.core.agent_runtime import resolve_tts_runtime
from app.core.config import settings
from app.core.error_responses import build_call_initiate_error_payload
from app.middleware.request_id_middleware import new_request_id
from app.core.logger import logger
from app.models.call_flow import CallFlow
from app.models.call_session import CallSession
from app.schemas.twilio import (
    CallInitiateRequest,
    CallInitiateResponse,
)
from app.schemas.base import SuccessResponse
from app.services.ab_testing_service import ab_testing_service
from app.services.agent_service import agent_service
from app.services.call_session_service import call_session_service
from app.services.credit_service import credit_service
from app.services.twilio_service import twilio_service
from app.services.voice_interview_context_service import (
    build_voice_interview_enrichment,
    parse_optional_uuid,
)
from app.utils.response import create_success_response
from app.routers.general_websocket import broadcast_call_status_update

# Outbound sessions occupying a workspace concurrent-call slot (must match DB values).
_ACTIVE_OUTBOUND_STATUSES = ("initiated", "ringing", "connected", "in-progress")

# `Agent.status` values that explicitly disqualify an agent from receiving
# outbound calls, regardless of whether a genuine active PhoneNumber binding
# exists (Check B, below). "inactive"/"draft" are the tenant's intentional
# disable states; "error" signals a telephony-provisioning failure. Any other
# status (including "active", which can legitimately coexist with a real
# binding due to the generic enable/disable PATCH path — see
# agent_service.py's _apply_ticket_update) is NOT treated as disqualifying:
# the real source of truth for callability is the PhoneNumber row itself.
_DISQUALIFYING_AGENT_STATUSES = {"inactive", "draft", "error"}

# Retry/backoff for transient Twilio call-creation failures — deliberately narrow:
# `calls.create` is a non-idempotent, side-effecting operation (it actually dials the
# customer) and Twilio provides no idempotency-key mechanism for it. Only HTTP 429 is
# retried, because Twilio guarantees a 429 means the request was rejected before being
# queued/dialed. A 5xx is ambiguous — it can mean the call was already dialed server-side
# before the error was returned (e.g. a gateway timeout) — so retrying on 5xx risks a real
# duplicate outbound call to the customer that would go untracked (only one
# call_session.twilio_call_sid is ever stored). 5xx / permanent 4xx errors (invalid number,
# unverified caller ID — Twilio error codes like 21211, 21214) all fail immediately.
# Mirrors the retry/backoff *pattern* used for CRM outbound HTTP calls elsewhere in this
# codebase (see ghl_service.py / hubspot_service.py _request_with_backoff), which retry
# only on 429 for the same "don't double-fire a side-effecting request" reasoning.
_TWILIO_CALL_MAX_RETRIES = 2
_TWILIO_CALL_BASE_BACKOFF_SECONDS = 0.5


def _is_transient_twilio_error(exc: TwilioRestException) -> bool:
    """True only for HTTP 429 (rate limited — Twilio guarantees the call was never
    dialed). False for everything else, including 5xx: `calls.create` is
    non-idempotent and side-effecting, so an ambiguous 5xx must not be retried
    (risk of dialing the customer twice)."""
    status_code = getattr(exc, "status", None)
    return status_code == 429


async def _create_twilio_call_with_retry(call_fn, **kwargs):
    """
    Call `twilio_service.make_call` / `make_call_with_credentials` with exponential
    backoff retry on transient failures only. Permanent errors raise immediately on
    the first attempt so the caller can fail fast.
    """
    attempt = 0
    while True:
        try:
            return call_fn(**kwargs)
        except TwilioRestException as exc:
            if not _is_transient_twilio_error(exc) or attempt >= _TWILIO_CALL_MAX_RETRIES:
                raise
            wait_seconds = _TWILIO_CALL_BASE_BACKOFF_SECONDS * (2 ** attempt)
            logger.warning(
                "Transient Twilio call-creation error (status=%s code=%s); retrying in %.1fs (attempt %d/%d)",
                exc.status,
                getattr(exc, "code", None),
                wait_seconds,
                attempt + 1,
                _TWILIO_CALL_MAX_RETRIES,
            )
            await asyncio.sleep(wait_seconds)
            attempt += 1


async def initiate_call(
    call_request: CallInitiateRequest,
    db: Session,
    is_system_call: bool,
    tenant_id: uuid.UUID | None,
    user_id: uuid.UUID | None,
    request_id: str | None = None,
) -> SuccessResponse[CallInitiateResponse] | JSONResponse:
    """
    Outbound call dispatch: LiveKit room → DB record → Twilio call.

    Sequence (per ticket BE1-S3):
      1. Resolve caller identity (auth already handled by the caller)
      2. Validate agent membership in workspace
      3. Check agent.status == "ready" (phone bound)
      4. Validate E.164 on toNumber (schema already enforces this)
      5. Validate fromNumber body param matches agent-bound number
      6. Credit check
      7. Per-workspace concurrent outbound limit
      8. Create LiveKit room (FAIL FAST — no side effects on error)
      9. Create call_session DB record with pre-assigned UUID
      10. Initiate Twilio call
      11. Return { callId, status: "initiated" }
    """
    request_id = request_id or new_request_id()

    def _err(http_status: int, error_code: str, message: str) -> JSONResponse:
        return JSONResponse(
            status_code=http_status,
            content=build_call_initiate_error_payload(
                http_status,
                message,
                call_request,
                error_code=error_code,
                request_id=request_id,
            ),
            headers={"X-Request-ID": request_id},
        )

    try:
        # ── 1. Identity resolution (auth already verified by caller) ──────
        if tenant_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required: JWT token or webhook secret",
            )
        tenant_id_filter = tenant_id
        user_id_filter = user_id

        # ── 2. Validate agent ─────────────────────────────────────────────
        try:
            agent_id = uuid.UUID(call_request.agentId)
            agent = agent_service.get_agent_by_id(db, agent_id, tenant_id_filter)
        except (ValueError, HTTPException):
            raise HTTPException(
                status_code=404, detail=f"Agent {call_request.agentId} not found"
            )

        # ── 3. Agent not ready (GAP 3) ────────────────────────────────────
        # See _DISQUALIFYING_AGENT_STATUSES above: only explicit disable /
        # provisioning-failure states block a call here. `status == "ready"`
        # is deliberately NOT required — Check B below (a genuine active
        # PhoneNumber row) is the real source of truth for callability.
        if agent.status in _DISQUALIFYING_AGENT_STATUSES:
            return _err(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "agent_not_ready",
                "Bind a phone number to this agent first.",
            )

        # ── 4. Phone number binding ───────────────────────────────────────
        from app.models.phone_number import PhoneNumber

        phone_number_obj = (
            db.query(PhoneNumber)
            .filter(
                PhoneNumber.assistant_id == agent.id,
                PhoneNumber.tenant_id == tenant_id_filter,
                PhoneNumber.status == "active",
            )
            .first()
        )
        if not phone_number_obj:
            return _err(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "agent_not_ready",
                "Bind a phone number to this agent first.",
            )

        # Validate optional phone_number_id override
        if call_request.phone_number_id:
            try:
                requested_id = uuid.UUID(call_request.phone_number_id)
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid phone_number_id format: {str(e)}",
                )
            if requested_id != phone_number_obj.id:
                # System/batch calls may dial out on a rotated number (the agent's
                # bound number was spam-flagged) rather than the bound number —
                # bypass the strict agent-binding restriction as long as the
                # requested number belongs to this tenant and is active.
                rotated_number_obj = None
                if is_system_call:
                    from sqlalchemy import or_ as _or_

                    rotated_number_obj = (
                        db.query(PhoneNumber)
                        .filter(
                            PhoneNumber.id == requested_id,
                            PhoneNumber.tenant_id == tenant_id_filter,
                            PhoneNumber.status == "active",
                            _or_(
                                PhoneNumber.assistant_id.is_(None),
                                PhoneNumber.assistant_id == agent.id,
                            ),
                        )
                        .first()
                    )
                if rotated_number_obj is None:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=(
                            "phone_number_id must be the phone number bound to this agent "
                            f"(bound id: {phone_number_obj.id})."
                        ),
                    )
                phone_number_obj = rotated_number_obj

        # ── 5. Validate fromNumber body param (GAP 1) ────────────────────
        from_number = phone_number_obj.phone_number
        if call_request.fromNumber and call_request.fromNumber != from_number:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"fromNumber '{call_request.fromNumber}' does not match the agent's "
                    f"bound phone number '{from_number}'. "
                    "Remove fromNumber from the request or use the bound number."
                ),
            )

        # ── 6. Credit check ───────────────────────────────────────────────
        if not agent.model:
            raise HTTPException(
                status_code=400, detail="Agent does not have a model configured"
            )

        model_name = agent.model.model_name
        outbound_tts_runtime = resolve_tts_runtime(agent, db=db)
        tts_provider_slug = outbound_tts_runtime.adapter_slug
        has_sufficient, current_credits, required_credits, decline_reason = (
            credit_service.has_sufficient_credits(
                db=db,
                tenant_id=tenant_id_filter,
                model_name=model_name,
                tts_provider_slug=tts_provider_slug,
                is_byo_elevenlabs=outbound_tts_runtime.is_byo_elevenlabs,
            )
        )

        if not has_sufficient:
            logger.warning(
                "❌ Insufficient credits: %s < %s (reason=%s)",
                current_credits,
                required_credits,
                decline_reason,
            )
            # Don't put the numeric balance in the client-facing message: when
            # wallet sharing is on, `current_credits` is the PARENT tenant's
            # balance, not this (sub-account) caller's own — echoing it back
            # would leak the parent's live wallet balance to any sub-account
            # API caller. Full detail (including the balance) stays in the
            # log line above for internal debugging only.
            return _err(
                status.HTTP_402_PAYMENT_REQUIRED,
                "insufficient_credits",
                f"Insufficient credits to initiate call. Model: {model_name}. Reason: {decline_reason}",
            )

        logger.info(
            "✅ Credit check passed: %s credits available for model %s (tts=%s)",
            current_credits,
            model_name,
            tts_provider_slug,
        )

        # ── 7. Per-workspace concurrent outbound limit (GAP 6) ────────────
        concurrent_count = (
            db.query(func.count(CallSession.id))
            .filter(
                CallSession.tenant_id == tenant_id_filter,
                CallSession.call_type == "outbound",
                CallSession.status.in_(_ACTIVE_OUTBOUND_STATUSES),
            )
            .scalar()
        ) or 0

        max_concurrent = settings.OUTBOUND_MAX_CONCURRENT_PER_WORKSPACE
        if concurrent_count >= max_concurrent:
            logger.warning(
                "Outbound concurrent limit reached for tenant %s: %d/%d",
                tenant_id_filter,
                concurrent_count,
                max_concurrent,
            )
            return _err(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "outbound_concurrent_limit_exceeded",
                f"Maximum concurrent outbound calls ({max_concurrent}) reached for this workspace. "
                "Wait for an active call to complete before placing a new one.",
            )

        # ── Resolve optional callFlowId so we can pass to LiveKit ────────
        flow_uuid: uuid.UUID | None = None
        requested_flow: CallFlow | None = None
        if call_request.callFlowId:
            try:
                flow_uuid = uuid.UUID(call_request.callFlowId)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid callFlowId format: must be a valid UUID",
                )

            requested_flow = db.execute(
                select(CallFlow).where(
                    CallFlow.id == flow_uuid,
                    CallFlow.tenant_id == tenant_id_filter,
                    CallFlow.is_deleted == False,  # noqa: E712
                )
            ).scalar_one_or_none()
            if requested_flow is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Call flow {flow_uuid} not found in workspace",
                )
            if requested_flow.status != "active":
                return _err(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    "call_flow_inactive",
                    f"Call flow {flow_uuid} is inactive and cannot be used to initiate calls. "
                    "Set it to active first.",
                )

        # ── Resolve Twilio credentials upfront ───────────────────────────
        use_custom_credentials = False
        account_sid: str | None = None
        auth_token: str | None = None

        if phone_number_obj.twilio_account_sid and phone_number_obj.twilio_auth_token:
            from app.core.security import decrypt_api_key

            account_sid = decrypt_api_key(phone_number_obj.twilio_account_sid)
            auth_token = decrypt_api_key(phone_number_obj.twilio_auth_token)
            use_custom_credentials = True
            logger.info(
                "Using bound number %s with per-number Twilio credentials",
                from_number,
            )
        else:
            from app.core.secret_manager import get_twilio_credentials

            account_sid, auth_token = get_twilio_credentials()
            use_custom_credentials = True
            logger.info(
                "Using bound number %s with platform Twilio credentials",
                from_number,
            )

        base_url = settings.WEBHOOK_BASE_URL

        # ── 8. Pre-generate call session UUID (ticket order: LiveKit → DB → Twilio) ──
        session_id = uuid.uuid4()

        # ── 8a. Create LiveKit room FIRST — fail fast before any side effects ─
        lk_meta: dict | None = None
        if settings.LIVEKIT_ENABLED:
            from app.services.livekit_service import livekit_service

            room_name = f"room_{session_id}"
            try:
                room = await livekit_service.create_room(
                    call_id=session_id,
                    agent_id=agent.id,
                    flow_id=flow_uuid,
                )
                agent_token = livekit_service.generate_agent_token(room_name)
                lk_meta = {
                    "room_name": room_name,
                    "room_sid": room.sid,
                    "agent_token": agent_token,  # never logged — secret
                    "flow_id": str(flow_uuid) if flow_uuid else None,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                logger.info(
                    "LiveKit room provisioned: %s (sid=%s) for session %s",
                    room_name,
                    room.sid,
                    session_id,
                )
            except Exception as exc:
                logger.error(
                    "LiveKit room creation failed for pre-assigned session %s: %s",
                    session_id,
                    exc,
                    exc_info=True,
                )
                # Attempt cleanup in case the room was partially created
                try:
                    await livekit_service.close_room(session_id)
                except Exception:  # noqa: S110 - best-effort cleanup after the room-creation failure already logged above
                    pass
                # Do NOT create DB record; do NOT call Twilio
                return _err(
                    status.HTTP_503_SERVICE_UNAVAILABLE,
                    "livekit_room_creation_failed",
                    "LiveKit room creation failed. Cannot initiate call without media transport.",
                )

        # ── 9. Create call_session DB record (after LiveKit succeeds) ────
        call_session = call_session_service.create_call_session(
            db=db,
            user_id=user_id_filter,
            agent_id=agent.id,
            tenant_id=tenant_id_filter,
            twilio_call_sid="",  # updated after Twilio call
            from_number=from_number,
            to_number=call_request.toNumber,
            call_type="outbound",
            session_id=session_id,
            status="initiated",
        )

        # Persist LiveKit metadata alongside session
        if lk_meta:
            call_session.call_metadata = {
                **(call_session.call_metadata or {}),
                "livekit": lk_meta,
            }
            db.commit()

        # LiveKit egress recording when enabled for this number (Twilio record stays off).
        _livekit_recording_enabled = False
        if settings.LIVEKIT_ENABLED and lk_meta:
            try:
                from app.models.phone_number import NumberConfiguration, PhoneNumber
                from sqlalchemy import select as _select

                _pn_stmt = _select(NumberConfiguration).join(
                    PhoneNumber, NumberConfiguration.phone_number_id == PhoneNumber.id
                ).where(
                    PhoneNumber.phone_number == from_number,
                    PhoneNumber.tenant_id == tenant_id_filter,
                )
                _nc = db.execute(_pn_stmt).scalar_one_or_none()
                _livekit_recording_enabled = bool(_nc and _nc.recording_enabled)
                if _livekit_recording_enabled:
                    from app.services.s3_recording_service import build_s3_key
                    from app.services.livekit_recording_service import livekit_recording_service

                    _gcs_path = build_s3_key(
                        workspace_id=tenant_id_filter,
                        call_id=session_id,
                    )
                    _session_id = session_id

                    async def _start_rec() -> None:
                        from app.db.session import SessionLocal

                        egress_id = await livekit_recording_service.start_room_recording(
                            call_id=_session_id,
                            workspace_id=tenant_id_filter,
                            gcs_path=_gcs_path,
                        )
                        if not egress_id:
                            return
                        _db = SessionLocal()
                        try:
                            _cs = _db.get(CallSession, _session_id)
                            if _cs is None:
                                return
                            _meta = dict(_cs.call_metadata or {})
                            _meta["recording"] = {
                                "egress_id": egress_id,
                                "gcs_path": _gcs_path,
                            }
                            _cs.call_metadata = _meta
                            _db.commit()
                            logger.info(
                                "Outbound LiveKit recording started: session=%s egress_id=%s",
                                _session_id,
                                egress_id,
                            )
                        finally:
                            _db.close()

                    asyncio.create_task(_start_rec())
            except Exception as _rec_exc:
                logger.warning("Outbound recording setup failed: %s", _rec_exc)

        # Optional appointment_id
        appt_raw = (call_request.appointment_id or "").strip()
        if appt_raw:
            md = {**(call_session.call_metadata or {})}
            md["appointment_id"] = appt_raw
            call_session.call_metadata = md
            db.commit()
            db.refresh(call_session)

        # Batch call — store substituted prompt for the voice agent runtime
        batch_record_id = (call_request.batch_call_record_id or "").strip()
        if batch_record_id or call_request.batch_prompt_override:
            md = {**(call_session.call_metadata or {})}
            if batch_record_id:
                md["batch_call_record_id"] = batch_record_id
            if call_request.batch_prompt_override:
                md["batch_prompt_override"] = call_request.batch_prompt_override
            call_session.call_metadata = md
            db.commit()
            db.refresh(call_session)

        # callFlowId
        if flow_uuid:
            call_session.call_flow_id = flow_uuid
            db.commit()
            db.refresh(call_session)

            # A/B prompt testing: assign + persist the variant now, before any
            # LLM request, so it's known even if the call fails mid-way. Locked
            # for the duration of the call via call_metadata["ab_prompt_text"].
            # Reuse the row already fetched by the active-status gate above —
            # intervening db.commit() calls expire it, so SQLAlchemy transparently
            # reloads fresh attributes on next access; no extra explicit SELECT needed.
            ab_testing_service.assign_and_lock_variant(
                db, call_session, requested_flow
            )

        # JD / resume enrichment (non-blocking on failure)
        _ctx = call_request.jd_context or {}
        _jd = parse_optional_uuid(
            call_request.jd_id
            or (
                str(_ctx.get("jd_id"))
                if _ctx.get("jd_id") is not None and str(_ctx.get("jd_id")).strip()
                else None
            )
        )
        _resume = parse_optional_uuid(
            call_request.resume_id
            or (
                str(_ctx.get("resume_id"))
                if _ctx.get("resume_id") is not None and str(_ctx.get("resume_id")).strip()
                else None
            )
        )
        if call_request.jd_context or _jd or _resume:
            try:
                enrich = build_voice_interview_enrichment(
                    db,
                    tenant_id=tenant_id_filter,
                    jd_id=_jd,
                    resume_id=_resume,
                    existing_jd_context=call_request.jd_context,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Voice interview enrichment skipped: %s", exc, exc_info=True)
                enrich = {
                    "merged_jd_context": {**(call_request.jd_context or {})},
                    "voice_dynamic_context": None,
                }
            md_enrich: dict = {**(call_session.call_metadata or {})}
            if enrich.get("merged_jd_context"):
                md_enrich["jd_context"] = enrich["merged_jd_context"]
            if enrich.get("voice_dynamic_context"):
                md_enrich["voice_dynamic_context"] = enrich["voice_dynamic_context"]
            call_session.call_metadata = md_enrich
            db.commit()
            db.refresh(call_session)

        # Webhook URLs
        streaming_webhook_url = (
            f"{base_url}/api/v1/voice/gather/streaming?"
            f"agentId={agent.id}&userId={user_id_filter}&callSessionId={call_session.id}"
        )
        status_callback_url = (
            f"{base_url}/api/v1/voice/webhook/call-events?"
            f"agentId={agent.id}&userId={user_id_filter}&callSessionId={call_session.id}"
        )

        # Answering Machine Detection (batch calls only) — hold the call on a
        # pause/redirect loop until the async AMD callback resolves human vs machine.
        amd_status_callback_url: str | None = None
        if call_request.enable_amd:
            batch_record_id_q = call_request.batch_call_record_id or ""
            amd_status_callback_url = (
                f"{base_url}/api/v1/webhooks/twilio/amd?"
                f"callSessionId={call_session.id}&batchCallRecordId={batch_record_id_q}"
            )
            webhook_url = (
                f"{base_url}/api/v1/webhooks/twilio/amd-hold?"
                f"agentId={agent.id}&userId={user_id_filter}&callSessionId={call_session.id}"
            )
        else:
            webhook_url = streaming_webhook_url

        logger.info("Making call with webhook_url: %s", webhook_url)
        logger.info("Making call with status_callback_url: %s", status_callback_url)

        # Optional WebSocket broadcast before dialling
        try:
            await broadcast_call_status_update(
                call_session_id=str(call_session.id),
                status="initiating",
                metadata={
                    "agent_id": str(agent.id),
                    "agent_name": agent.name,
                    "to_number": call_request.toNumber,
                    "from_number": from_number,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as e:  # pragma: no cover - non-critical
            logger.warning("⚠️ WebSocket broadcast failed (non-critical): %s", e)

        # ── 10. Initiate Twilio call ──────────────────────────────────────
        _twilio_record = not _livekit_recording_enabled
        try:
            amd_kwargs = {}
            if requested_flow and requested_flow.voicemail_detection_enabled:
                detection_type = (
                    "DetectMessageEnd"
                    if requested_flow.voicemail_advanced_detection_enabled
                    else "Enable"
                )
                timeout_val = requested_flow.voicemail_detection_timeout or 5
                amd_kwargs = {
                    "machine_detection": detection_type,
                    "machine_detection_timeout": timeout_val,
                    "async_amd": "true",
                    "async_amd_status_callback": amd_status_callback_url,
                }
            elif call_request.enable_amd:
                amd_kwargs = {
                    "machine_detection": "Enable",
                    "machine_detection_timeout": 3,
                    "async_amd": "true",
                    "async_amd_status_callback": amd_status_callback_url,
                }
            if use_custom_credentials:
                call = await _create_twilio_call_with_retry(
                    twilio_service.make_call_with_credentials,
                    to_number=call_request.toNumber,
                    from_number=from_number,
                    webhook_url=webhook_url,
                    status_callback_url=status_callback_url,
                    account_sid=account_sid,
                    auth_token=auth_token,
                    record=_twilio_record,
                    **amd_kwargs,
                )
            else:
                call = await _create_twilio_call_with_retry(
                    twilio_service.make_call,
                    to_number=call_request.toNumber,
                    from_number=from_number,
                    webhook_url=webhook_url,
                    status_callback_url=status_callback_url,
                    record=_twilio_record,
                    **amd_kwargs,
                )
        except Exception as exc:
            logger.error(
                "Twilio call creation failed for session %s: %s",
                call_session.id,
                exc,
                exc_info=True,
            )
            call_session.status = "failed"
            call_session.ended_reason = "Call.start.error"
            db.commit()
            if settings.LIVEKIT_ENABLED and lk_meta:
                try:
                    from app.services.livekit_service import livekit_service

                    await livekit_service.close_room(call_session.id)
                except Exception:  # noqa: S110 - best-effort cleanup after the Twilio failure already logged above
                    pass
            return _err(
                status.HTTP_502_BAD_GATEWAY,
                "twilio_call_failed",
                "Failed to initiate phone call via Twilio.",
            )
        logger.info("✅ Call initiated successfully")

        # Update call session with Twilio SID
        call_session.twilio_call_sid = call.sid
        db.commit()
        logger.info(
            "✅ Updated call session %s with Twilio SID: %s",
            call_session.id,
            call.sid,
        )

        # Broadcast call initiated event after Twilio confirms
        try:
            await broadcast_call_status_update(
                call_session_id=str(call_session.id),
                status="initiated",
                metadata={
                    "call_sid": call.sid,
                    "agent_id": str(agent.id),
                    "agent_name": agent.name,
                    "to_number": call_request.toNumber,
                    "from_number": from_number,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as e:  # pragma: no cover - non-critical
            logger.warning("⚠️ Failed to send call initiated event: %s", e)

        call_id = f"call_{call.sid[-8:]}"

        crm_container_id = call_request.crm_container_id or call_request.board_id
        crm_item_id = call_request.crm_item_id or call_request.monday_item_id
        status_field_id = call_request.status_field_id or call_request.status_column_id
        call_session_id_field_id = (
            call_request.call_session_id_field_id
            or call_request.call_session_id_column_id
        )

        return create_success_response(
            CallInitiateResponse(
                callId=call_id,
                twilioCallSid=call.sid,
                callSessionId=str(call_session.id),
                status="initiated",
                board_id=call_request.board_id,
                monday_item_id=call_request.monday_item_id,
                status_column_id=call_request.status_column_id,
                call_session_id_column_id=call_request.call_session_id_column_id,
                crm_container_id=crm_container_id,
                crm_item_id=crm_item_id,
                status_field_id=status_field_id,
                call_session_id_field_id=call_session_id_field_id,
                crm_type=call_request.crm_type,
            ),
            "Call initiated successfully",
        )

    except HTTPException as e:
        logger.warning("Call initiate HTTP error: %s", e.detail)
        return JSONResponse(
            status_code=e.status_code,
            content=build_call_initiate_error_payload(
                e.status_code, e.detail, call_request, request_id=request_id
            ),
            headers={"X-Request-ID": request_id},
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.error("Call initiate failed: %s", e, exc_info=True)
        return JSONResponse(
            status_code=500,
            content=build_call_initiate_error_payload(
                500, None, call_request, request_id=request_id
            ),
            headers={"X-Request-ID": request_id},
        )
