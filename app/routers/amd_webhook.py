"""
Twilio Answering Machine Detection (AMD) webhooks — batch outbound calls only.

POST /amd       — async AMD status callback (AnsweredBy: human | machine_start |
                   machine_end_beep | machine_end_silence | fax | unknown)
POST /amd-hold   — TwiML the call answers into while AMD is still resolving;
                    pauses/redirects itself until the callback above records a result,
                    then redirects into the normal streaming webhook (human/unknown/continue)
                    or is torn down by the callback (machine + skip/leave_message).

Both routes are exempted from x-api-key/JWT auth (see api_key_middleware._SKIP_PREFIXES)
since Twilio calls them directly — authenticated instead via X-Twilio-Signature.

Docs: https://www.twilio.com/docs/voice/answering-machine-detection#async-amd
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session, joinedload
from twilio.twiml.voice_response import VoiceResponse

from app.api.deps import get_db
from app.core.config import settings
from app.core.logger import logger
from app.models.batch_call_record import BatchCallRecord
from app.models.batch_job import BatchJob
from app.services.batch_call_worker_service import BatchCallWorkerService
from app.services.call_session_service import call_session_service
from app.services.twilio_service import twilio_service
from app.services.voice_interview_context_service import parse_optional_uuid
from app.utils.twilio_validation import (
    validate_twilio_signature,
    validate_twilio_signature_with_token,
)

router = APIRouter(tags=["AMD Webhook"])

# amd_result values that release the amd-hold loop into the normal call flow.
_HOLD_RELEASE_RESULTS = ("human", "unknown", "continue")

# Give up holding after this long (covers a dropped/late async AMD callback)
# and let the call proceed into the normal streaming flow rather than pause forever.
_MAX_HOLD_SECONDS = 20


def _resolve_credentials(
    db: Session, call_session
) -> tuple[Optional[str], Optional[str]]:
    """Best-effort per-tenant Twilio credentials; falls back to global creds inside twilio_service."""
    if call_session is None:
        return None, None
    try:
        from app.utils.voice_twilio_utils import get_twilio_credentials_for_call

        return get_twilio_credentials_for_call(db, call_session)
    except Exception as exc:
        logger.warning(
            "AMD webhook: per-session Twilio credentials unavailable (%s); falling back to global",
            exc,
        )
        return None, None


async def _validate_amd_signature(
    request: Request, db: Session, call_session, form: dict
) -> bool:
    if settings.ALLOW_UNAUTHENTICATED_WEBHOOKS:
        return True
    _, auth_token = _resolve_credentials(db, call_session)
    if auth_token and validate_twilio_signature_with_token(request, form, auth_token):
        return True
    return validate_twilio_signature(request, form)


def _streaming_url(
    agentId: Optional[str], userId: Optional[str], callSessionId: Optional[str]
) -> str:
    return (
        f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/gather/streaming?"
        f"agentId={agentId}&userId={userId}&callSessionId={callSessionId}"
    )


def _amd_hold_url(
    agentId: Optional[str], userId: Optional[str], callSessionId: Optional[str]
) -> str:
    return (
        f"{settings.WEBHOOK_BASE_URL}/api/v1/webhooks/twilio/amd-hold?"
        f"agentId={agentId}&userId={userId}&callSessionId={callSessionId}"
    )


@router.post("/amd-hold", response_class=HTMLResponse, include_in_schema=False)
async def amd_hold(
    request: Request,
    agentId: Optional[str] = None,
    userId: Optional[str] = None,
    callSessionId: Optional[str] = None,
    db: Session = Depends(get_db),
) -> HTMLResponse:
    """Keep the call alive (pause + redirect loop) until the async AMD callback resolves."""
    session_uuid = parse_optional_uuid(callSessionId)
    call_session = (
        call_session_service.get_call_session_by_id(db, session_uuid)
        if session_uuid
        else None
    )

    form = dict(await request.form())
    if not await _validate_amd_signature(request, db, call_session, form):
        logger.warning(
            "AMD hold: invalid Twilio signature for callSessionId=%s", callSessionId
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid Twilio signature"
        )

    vr = VoiceResponse()

    if call_session is None:
        # Can't resolve the session — fail open into the normal flow instead of
        # holding the call indefinitely.
        vr.redirect(_streaming_url(agentId, userId, callSessionId))
        return HTMLResponse(str(vr), media_type="application/xml")

    amd_result = (call_session.call_metadata or {}).get("amd_result")
    start_time = call_session.start_time
    if start_time is not None and start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    elapsed_seconds = (
        (datetime.now(timezone.utc) - start_time).total_seconds() if start_time else 0
    )

    if amd_result in _HOLD_RELEASE_RESULTS or elapsed_seconds > _MAX_HOLD_SECONDS:
        vr.redirect(_streaming_url(agentId, userId, callSessionId))
    else:
        # Still waiting on AMD (or it resolved to a machine — the AMD callback
        # hangs up / injects TwiML directly via the REST API in that case).
        vr.pause(length=1)
        vr.redirect(_amd_hold_url(agentId, userId, callSessionId))

    return HTMLResponse(str(vr), media_type="application/xml")


@router.post("/amd", response_class=HTMLResponse)
async def amd_callback(
    request: Request,
    callSessionId: Optional[str] = None,
    batchCallRecordId: Optional[str] = None,
    db: Session = Depends(get_db),
) -> HTMLResponse:
    """Twilio async AMD status callback — see module docstring."""
    form = dict(await request.form())
    answered_by = (form.get("AnsweredBy") or "").strip()
    call_sid = form.get("CallSid")

    session_uuid = parse_optional_uuid(callSessionId)
    call_session = (
        call_session_service.get_call_session_by_id(db, session_uuid)
        if session_uuid
        else None
    )

    if not await _validate_amd_signature(request, db, call_session, form):
        logger.warning(
            "AMD callback: invalid Twilio signature for callSessionId=%s", callSessionId
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid Twilio signature"
        )

    record_uuid = parse_optional_uuid(batchCallRecordId)
    record: Optional[BatchCallRecord] = None
    job: Optional[BatchJob] = None
    if record_uuid:
        record = (
            db.query(BatchCallRecord)
            .options(joinedload(BatchCallRecord.batch_job))
            .filter(BatchCallRecord.id == record_uuid)
            .first()
        )
        job = record.batch_job if record else None

    logger.info(
        "AMD callback: callSessionId=%s batchCallRecordId=%s AnsweredBy=%s",
        callSessionId,
        batchCallRecordId,
        answered_by,
    )

    # Missing job (e.g. a non-batch call that set enable_amd, or a stale record)
    # fails open into "continue" rather than silently hanging up a real caller.
    voicemail_action = job.voicemail_action if job else "continue"
    worker_svc = BatchCallWorkerService(db)

    if answered_by == "machine_start":
        if voicemail_action == "skip":
            # Commit the DB transition first so a concurrent Twilio call-status
            # webhook (fired once the REST hangup below lands) sees the record
            # already off "active" instead of racing to double-count/bill it.
            if record is not None:
                await worker_svc.mark_voicemail_skipped(record.id)
            _persist_amd_result(db, call_session, "machine_start")
            if call_sid:
                _hangup(db, call_session, call_sid)
        elif voicemail_action == "leave_message":
            _persist_amd_result(db, call_session, "machine_start")
            # nothing else to do yet — wait for machine_end_beep below.
        else:  # "continue" (or no job resolved) — let the normal flow proceed
            _persist_amd_result(db, call_session, "continue")

    elif answered_by == "machine_end_beep":
        if voicemail_action == "leave_message" and call_sid:
            message = (job.voicemail_message or "").strip() if job else ""
            if message:
                if record is not None:
                    await worker_svc.mark_voicemail_message_left(record.id)
                _play_voicemail_and_hangup(db, call_session, call_sid, message)
            else:
                # No message configured — nothing to deliver, treat as a skip.
                if record is not None:
                    await worker_svc.mark_voicemail_skipped(record.id)
                _hangup(db, call_session, call_sid)

    elif answered_by in ("human", "unknown"):
        _persist_amd_result(db, call_session, answered_by)

    # "fax" / anything else: no special handling — let the call proceed/expire naturally.

    return HTMLResponse("", media_type="application/xml")


def _persist_amd_result(db: Session, call_session, result: str) -> None:
    if call_session is None:
        return
    md = {**(call_session.call_metadata or {})}
    md["amd_result"] = result
    call_session.call_metadata = md
    db.commit()


def _hangup(db: Session, call_session, call_sid: str) -> None:
    account_sid, auth_token = _resolve_credentials(db, call_session)
    ok = (
        twilio_service.end_call_with_credentials(call_sid, account_sid, auth_token)
        if account_sid and auth_token
        else twilio_service.end_call(call_sid)
    )
    if not ok:
        logger.warning("AMD webhook: failed to hang up call_sid=%s", call_sid)


def _play_voicemail_and_hangup(
    db: Session, call_session, call_sid: str, message: str
) -> None:
    lang = "en"
    voice = "female"
    if call_session is not None and call_session.agent_id is not None:
        from app.models.agent import Agent

        agent = db.get(Agent, call_session.agent_id)
        if agent is not None:
            lang = agent.language or lang
            voice = agent.voice_type or voice

    vr = VoiceResponse()
    tts_url = (
        f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?"
        f"text={quote(message)}&lang={lang}&voice={voice}"
    )
    vr.play(tts_url)
    vr.hangup()

    account_sid, auth_token = _resolve_credentials(db, call_session)
    ok = twilio_service.update_call_twiml(call_sid, str(vr), account_sid, auth_token)
    if not ok:
        logger.warning(
            "AMD webhook: failed to play voicemail message for call_sid=%s", call_sid
        )
