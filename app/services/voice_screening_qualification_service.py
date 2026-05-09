"""
When a voice screening call ends with [SCREENING_QUALIFIED], mark the linked resume qualified.

Uses call_session.call_metadata[\"jd_context\"][\"resume_id\"] set during outbound initiate.

JD recruitment calls are flagged with jd_context[\"recruitment_jd_screening\"] == True when a job
description is resolved (voice_interview_context_service). Generic agent flows omit this flag.
"""
from __future__ import annotations

from datetime import datetime, timezone

import uuid

from sqlalchemy.orm import Session

from app.core.logger import logger
from app.models.call_session import CallSession
from app.models.resume import CandidateStatus, Resume


def is_jd_recruitment_voice_context(jd_ctx: dict | None) -> bool:
    """
    True only for outbound JD/recruitment screening (job row resolved), so booking/general calls
    are unaffected even if the model echoes screening tokens.

    Prefer explicit recruitment_jd_screening from enrichment; fallback for older sessions:
    both jd_id and resume_id present on jd_context.
    """
    if not isinstance(jd_ctx, dict):
        return False
    if jd_ctx.get("recruitment_jd_screening") is True:
        return True
    rid = jd_ctx.get("resume_id")
    jid = jd_ctx.get("jd_id")
    if rid and str(rid).strip() and jid and str(jid).strip():
        return True
    # Enrichment may set jd_title + resume_id when JD resolved; allow qualify path for same session
    if rid and str(rid).strip() and jd_ctx.get("jd_title") and str(jd_ctx.get("jd_title")).strip():
        return True
    return False


_SIGNAL_META_KEY = "voice_screening_qualified_signal"


def persist_voice_screening_qualified_signal(db: Session, call_session: CallSession | None) -> None:
    """Persist intent when LLM emits success tokens (transcript strips them). Webhook can finish qualify."""
    if not call_session:
        return
    try:
        md = dict(call_session.call_metadata or {})
        md[_SIGNAL_META_KEY] = True
        call_session.call_metadata = md
        db.add(call_session)
        db.commit()
        db.refresh(call_session)
    except Exception as exc:
        logger.warning("persist_voice_screening_qualified_signal failed: %s", exc, exc_info=True)


def maybe_qualify_resume_on_call_completed(db: Session, call_session_id: uuid.UUID) -> bool:
    """
    Twilio 'completed' webhook fallback: stream may crash before DB qualify runs; we also persist
    voice_screening_qualified_signal on metadata when the LLM emits success tokens (transcript strips those tokens).
    """
    cs = db.query(CallSession).filter(CallSession.id == call_session_id).first()
    if not cs:
        return False
    try:
        db.refresh(cs)
    except Exception:
        pass
    md = cs.call_metadata if isinstance(cs.call_metadata, dict) else {}
    if not md.get(_SIGNAL_META_KEY):
        return False
    return apply_resume_qualified_after_voice_screening(db, cs)


def apply_resume_qualified_after_voice_screening(db: Session, call_session: CallSession | None) -> bool:
    if not call_session:
        return False
    try:
        db.refresh(call_session)
    except Exception:
        pass
    md = call_session.call_metadata
    if not isinstance(md, dict):
        logger.debug(
            "Voice screening qualify: no call_metadata dict (session=%s)", call_session.id
        )
        return False
    jd_ctx = md.get("jd_context")
    if not isinstance(jd_ctx, dict):
        logger.debug(
            "Voice screening qualify: no jd_context on session %s — outbound call needs resume/JD initiate enrich",
            call_session.id,
        )
        return False
    rid_raw = jd_ctx.get("resume_id")
    if rid_raw is None or not str(rid_raw).strip():
        return False
    try:
        resume_uuid = uuid.UUID(str(rid_raw).strip())
    except (ValueError, TypeError):
        logger.warning(
            "Voice screening qualify: invalid resume_id in jd_context for session %s",
            call_session.id,
        )
        return False

    if not is_jd_recruitment_voice_context(jd_ctx):
        logger.debug(
            "Voice screening qualify: skipped — not a JD recruitment call (session=%s)",
            call_session.id,
        )
        return False

    tenant_id = call_session.tenant_id
    resume = (
        db.query(Resume)
        .filter(Resume.id == resume_uuid, Resume.tenant_id == tenant_id)
        .first()
    )
    if not resume:
        logger.warning(
            "Voice screening qualify: resume %s not found for tenant %s (session %s)",
            resume_uuid,
            tenant_id,
            call_session.id,
        )
        return False

    if resume.candidate_status == CandidateStatus.QUALIFIED:
        md_session = dict(call_session.call_metadata or {})
        md_session.pop(_SIGNAL_META_KEY, None)
        call_session.call_metadata = md_session
        db.add(call_session)
        db.commit()
        logger.debug(
            "Voice screening qualify: resume %s already qualified (session=%s)",
            resume_uuid,
            call_session.id,
        )
        return True

    resume.candidate_status = CandidateStatus.QUALIFIED
    resume.updated_at = datetime.now(timezone.utc)
    db.add(resume)
    md_session = dict(call_session.call_metadata or {})
    md_session.pop(_SIGNAL_META_KEY, None)
    call_session.call_metadata = md_session
    db.add(call_session)
    db.commit()
    logger.info(
        "Voice screening: marked resume %s qualified (call_session=%s)",
        resume_uuid,
        call_session.id,
    )
    return True
