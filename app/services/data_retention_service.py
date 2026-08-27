"""Data Retention Service — purges expired transcripts, summaries, and audio recordings
according to CallFlow retention policies while strictly preserving call session metadata.
"""

from __future__ import annotations

import datetime
import uuid

from fastapi import HTTPException, status
from sqlalchemy import delete, or_, select
from sqlalchemy.orm import Session

from app.core.logger import logger
from app.models.call_flow import CallFlow
from app.models.call_session import CallSession
from app.models.transcript_message import TranscriptMessage
from app.schemas.call_flow import DataRetentionPurgeResponse
from app.services import s3_recording_service


def purge_expired_call_data(
    db: Session,
    *,
    tenant_id: uuid.UUID,
    flow_id: uuid.UUID | None = None,
) -> DataRetentionPurgeResponse:
    """Purge expired transcripts, summaries, and recordings for calls matching active retention policies.

    If *flow_id* is provided, evaluates only that specific flow; otherwise processes all active flows
    with retention_policy_enabled=True under *tenant_id*.
    """
    now_aware = datetime.datetime.now(datetime.timezone.utc)
    now_naive = datetime.datetime.utcnow()

    if flow_id is not None:
        single_flow = db.execute(
            select(CallFlow).where(
                CallFlow.id == flow_id,
                CallFlow.tenant_id == tenant_id,
                CallFlow.is_deleted.is_(False),
            )
        ).scalar_one_or_none()
        if single_flow is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Call flow {flow_id} not found",
            )
        flows = [single_flow] if single_flow.retention_policy_enabled else []
    else:
        flow_query = select(CallFlow).where(
            CallFlow.tenant_id == tenant_id,
            CallFlow.is_deleted.is_(False),
            CallFlow.retention_policy_enabled.is_(True),
        )
        flows = list(db.execute(flow_query).scalars().all())

    purged_transcripts = 0
    purged_summaries = 0
    purged_recordings = 0
    affected_session_ids: set[uuid.UUID] = set()
    s3_paths_to_delete: list[str] = []

    for flow in flows:
        if not flow.retention_policy_enabled:
            continue

        # 1. Transcripts purge
        if flow.retention_transcript_enabled:
            t_days = flow.retention_transcript_days or 30
            t_cutoff_aware = now_aware - datetime.timedelta(days=t_days)
            t_cutoff_naive = now_naive - datetime.timedelta(days=t_days)
            t_sessions = (
                db.execute(
                    select(CallSession).where(
                        CallSession.call_flow_id == flow.id,
                        CallSession.tenant_id == tenant_id,
                        or_(
                            CallSession.start_time < t_cutoff_aware,
                            CallSession.start_time < t_cutoff_naive,
                        ),
                    )
                )
                .scalars()
                .all()
            )
            for sess in t_sessions:
                had_transcript = sess.call_transcript is not None
                del_res = db.execute(
                    delete(TranscriptMessage).where(
                        TranscriptMessage.call_session_id == sess.id
                    )
                )
                had_messages = bool(del_res.rowcount and del_res.rowcount > 0)
                if had_transcript or had_messages:
                    sess.call_transcript = None
                    affected_session_ids.add(sess.id)
                    purged_transcripts += 1

        # 2. Summaries purge
        if flow.retention_summary_enabled:
            s_days = flow.retention_summary_days or 30
            s_cutoff_aware = now_aware - datetime.timedelta(days=s_days)
            s_cutoff_naive = now_naive - datetime.timedelta(days=s_days)
            s_sessions = (
                db.execute(
                    select(CallSession).where(
                        CallSession.call_flow_id == flow.id,
                        CallSession.tenant_id == tenant_id,
                        or_(
                            CallSession.start_time < s_cutoff_aware,
                            CallSession.start_time < s_cutoff_naive,
                        ),
                        CallSession.transcript_summary.is_not(None),
                    )
                )
                .scalars()
                .all()
            )
            for sess in s_sessions:
                sess.transcript_summary = None
                affected_session_ids.add(sess.id)
                purged_summaries += 1

        # 3. Audio recordings purge
        if flow.retention_recording_enabled:
            r_days = flow.retention_recording_days or 30
            r_cutoff_aware = now_aware - datetime.timedelta(days=r_days)
            r_cutoff_naive = now_naive - datetime.timedelta(days=r_days)
            r_sessions = (
                db.execute(
                    select(CallSession).where(
                        CallSession.call_flow_id == flow.id,
                        CallSession.tenant_id == tenant_id,
                        or_(
                            CallSession.start_time < r_cutoff_aware,
                            CallSession.start_time < r_cutoff_naive,
                        ),
                        or_(
                            CallSession.recording_s3_path.is_not(None),
                            CallSession.recording_url.is_not(None),
                        ),
                    )
                )
                .scalars()
                .all()
            )
            for sess in r_sessions:
                if sess.recording_s3_path:
                    s3_paths_to_delete.append(sess.recording_s3_path)
                    sess.recording_s3_path = None
                sess.recording_url = None
                affected_session_ids.add(sess.id)
                purged_recordings += 1

    if affected_session_ids:
        # Commit database records first so DB state is consistent before deleting objects from S3
        db.commit()
        for s3_path in s3_paths_to_delete:
            try:
                s3_recording_service.delete_recording_object(s3_path)
            except Exception as s3_err:
                logger.warning(
                    "Failed to delete S3 recording object after retention commit %s: %s",
                    s3_path,
                    s3_err,
                )
        logger.info(
            "Data retention purge completed: flow=%s tenant=%s transcripts=%d summaries=%d recordings=%d sessions=%d",
            flow_id,
            tenant_id,
            purged_transcripts,
            purged_summaries,
            purged_recordings,
            len(affected_session_ids),
        )

    return DataRetentionPurgeResponse(
        flow_id=flow_id,
        tenant_id=tenant_id,
        purged_transcripts_count=purged_transcripts,
        purged_summaries_count=purged_summaries,
        purged_recordings_count=purged_recordings,
        purged_sessions_count=len(affected_session_ids),
        message=(
            f"Data retention purge completed: {purged_transcripts} transcripts, "
            f"{purged_summaries} summaries, {purged_recordings} recordings purged "
            f"across {len(affected_session_ids)} sessions."
        ),
    )
