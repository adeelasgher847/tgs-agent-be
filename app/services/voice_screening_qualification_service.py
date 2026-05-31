"""
Apply candidate status on JD voice screening completion.

Primary signal sources (stored on call metadata):
- [SCREENING_QUALIFIED] token -> qualified
- [OUTCOME: ...] token -> mapped to qualified / partially qualified / rejected

Uses call_session.call_metadata["jd_context"]["resume_id"] set during outbound initiate.
JD recruitment calls are flagged with jd_context["recruitment_jd_screening"] == True when a job
description is resolved (voice_interview_context_service). Generic agent flows omit this flag.
"""
from __future__ import annotations

from datetime import datetime, timezone
import re

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
_CANDIDATE_STATUS_META_KEY = "voice_screening_candidate_status"
_OUTCOME_RE = re.compile(r"\[\s*OUTCOME\s*:\s*([^\]]+)\]", re.IGNORECASE)


def _normalize_outcome_to_candidate_status(outcome: str | None) -> CandidateStatus | None:
    if not outcome:
        return None
    token = str(outcome).strip().lower()
    if not token:
        return None

    compact = token.replace("_", " ").replace("-", " ")
    if "success" in compact or "qualified" in compact or compact in {"pass", "yes"}:
        return CandidateStatus.QUALIFIED
    if (
        "reject" in compact
        or "fail" in compact
        or "not qualified" in compact
        or compact in {"no", "decline"}
    ):
        return CandidateStatus.REJECTED
    if "partial" in compact or "unclear" in compact or "maybe" in compact:
        return CandidateStatus.PARTIALLY_QUALIFIED
    return None


def _extract_status_signal_from_text(text: str | None) -> CandidateStatus | None:
    if not text:
        return None
    if "[SCREENING_QUALIFIED]" in text.upper():
        return CandidateStatus.QUALIFIED
    match = _OUTCOME_RE.search(text)
    if not match:
        return None
    return _normalize_outcome_to_candidate_status(match.group(1))


def persist_voice_screening_status_signal(
    db: Session,
    call_session: CallSession | None,
    full_response_text: str | None,
) -> CandidateStatus | None:
    """
    Persist candidate-status intent when control tokens are emitted in streamed LLM output.
    """
    if not call_session:
        return None
    status_signal = _extract_status_signal_from_text(full_response_text)
    if status_signal is None:
        return None
    try:
        md = dict(call_session.call_metadata or {})
        md[_CANDIDATE_STATUS_META_KEY] = status_signal.value
        if status_signal == CandidateStatus.QUALIFIED:
            md[_SIGNAL_META_KEY] = True
        call_session.call_metadata = md
        db.add(call_session)
        db.commit()
        db.refresh(call_session)
        logger.info(
            "voice_screening_candidate_status persisted=%s (session=%s)",
            status_signal.value,
            call_session.id,
        )
        return status_signal
    except Exception as exc:
        logger.warning("persist_voice_screening_status_signal failed: %s", exc, exc_info=True)
        return None


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
        logger.info(
            "voice_screening_qualified_signal persisted (session=%s)", call_session.id
        )
    except Exception as exc:
        logger.warning("persist_voice_screening_qualified_signal failed: %s", exc, exc_info=True)


def maybe_update_resume_status_on_call_completed(db: Session, call_session_id: uuid.UUID) -> bool:
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
    if not md.get(_SIGNAL_META_KEY) and not md.get(_CANDIDATE_STATUS_META_KEY):
        return False
    return apply_resume_candidate_status_after_voice_screening(db, cs)


def maybe_qualify_resume_on_call_completed(db: Session, call_session_id: uuid.UUID) -> bool:
    """
    Backward-compatible alias.
    """
    return maybe_update_resume_status_on_call_completed(db, call_session_id)


def apply_resume_candidate_status_after_voice_screening(
    db: Session,
    call_session: CallSession | None,
) -> bool:
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

    signal_status_raw = md.get(_CANDIDATE_STATUS_META_KEY)
    signal_status: CandidateStatus | None = None
    try:
        if signal_status_raw:
            signal_status = CandidateStatus(str(signal_status_raw).strip())
    except ValueError:
        signal_status = None
    target_status = signal_status or (
        CandidateStatus.QUALIFIED if md.get(_SIGNAL_META_KEY) else None
    )
    if target_status is None:
        return False

    if resume.candidate_status == target_status:
        md_session = dict(call_session.call_metadata or {})
        md_session.pop(_SIGNAL_META_KEY, None)
        md_session.pop(_CANDIDATE_STATUS_META_KEY, None)
        call_session.call_metadata = md_session
        db.add(call_session)
        db.commit()
        logger.debug(
            "Voice screening: resume %s already %s (session=%s)",
            resume_uuid,
            target_status.value,
            call_session.id,
        )
        return True

    resume.candidate_status = target_status
    resume.updated_at = datetime.now(timezone.utc)
    db.add(resume)
    md_session = dict(call_session.call_metadata or {})
    md_session.pop(_SIGNAL_META_KEY, None)
    md_session.pop(_CANDIDATE_STATUS_META_KEY, None)
    call_session.call_metadata = md_session
    db.add(call_session)
    db.commit()
    logger.info(
        "Voice screening: marked resume %s as %s (call_session=%s)",
        resume_uuid,
        target_status.value,
        call_session.id,
    )
    return True


def apply_resume_qualified_after_voice_screening(db: Session, call_session: CallSession | None) -> bool:
    """
    Backward-compatible alias.
    """
    return apply_resume_candidate_status_after_voice_screening(db, call_session)
