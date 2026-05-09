"""
When a voice screening call ends with [SCREENING_QUALIFIED], mark the linked resume qualified.

Uses call_session.call_metadata[\"jd_context\"][\"resume_id\"] set during outbound initiate.
"""
from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.core.logger import logger
from app.models.call_session import CallSession
from app.models.resume import CandidateStatus, Resume


def apply_resume_qualified_after_voice_screening(db: Session, call_session: CallSession | None) -> bool:
    if not call_session:
        return False
    md = call_session.call_metadata
    if not isinstance(md, dict):
        return False
    jd_ctx = md.get("jd_context")
    if not isinstance(jd_ctx, dict):
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
