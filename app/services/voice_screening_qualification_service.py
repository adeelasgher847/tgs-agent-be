"""
When a voice screening call ends with [SCREENING_QUALIFIED], mark the linked resume qualified.

Uses call_session.call_metadata[\"jd_context\"][\"resume_id\"] set during outbound initiate.

JD recruitment calls are flagged with jd_context[\"recruitment_jd_screening\"] == True when a job
description is resolved (voice_interview_context_service). Generic agent flows omit this flag.
"""
from __future__ import annotations

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
    jd_id = jd_ctx.get("jd_id")
    resume_id = jd_ctx.get("resume_id")
    return bool(
        jd_id and str(jd_id).strip() and resume_id and str(resume_id).strip()
    )


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

    resume.candidate_status = CandidateStatus.QUALIFIED
    db.add(resume)
    db.commit()
    logger.info(
        "Voice screening: marked resume %s qualified (call_session=%s)",
        resume_uuid,
        call_session.id,
    )
    return True
