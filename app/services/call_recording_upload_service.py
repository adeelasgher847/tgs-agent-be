"""
Call Recording Upload Service — post-call async S3 upload job.

Follows the same pattern as inbound_call_crm_sync_service:
  schedule_recording_upload() -> asyncio.create_task() -> run_in_executor()

Flow:
  1. Load call_session
  2. Skip if recording not enabled / already has a path / egress_id missing
  3. Check LiveKit egress status (COMPLETE)
  4. Update S3 object metadata (callId, workspaceId, agentId, duration)
  5. On success: set recording_s3_path, clear recording_error
  6. On failure: log error, set recording_error=True — no auto-retry in Sprint 2
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Optional

from app.core.logger import logger


def schedule_recording_upload(call_session_id: uuid.UUID) -> None:
    """
    Fire-and-forget: schedule GCS upload after call ends.

    Mirrors schedule_inbound_crm_sync() in inbound_call_crm_sync_service.
    Called from bidirectional_stream._full_shutdown() and
    voice.handle_call_events_webhook (status=completed).
    """
    asyncio.create_task(_upload_recording_task(call_session_id))


async def _upload_recording_task(call_session_id: uuid.UUID) -> None:
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _upload_recording_sync, call_session_id)
    except Exception as exc:
        logger.error(
            "Recording upload task failed for session %s: %s",
            call_session_id,
            exc,
            exc_info=True,
        )


def _upload_recording_sync(call_session_id: uuid.UUID) -> None:
    """
    Synchronous upload worker — runs in executor so it doesn't block the event loop.

    Uses its own DB session (same pattern as _post_call_appointment_sync).
    """
    from app.db.session import SessionLocal
    from app.models.call_session import CallSession
    from app.services.recording_config_service import get_recording_enabled_for_call

    db = SessionLocal()
    try:
        session: Optional[CallSession] = db.get(CallSession, call_session_id)
        if session is None:
            logger.warning("Recording upload: call_session %s not found", call_session_id)
            return

        # Skip if recording was not enabled for this call's number
        if not get_recording_enabled_for_call(db, session):
            logger.debug(
                "Recording upload: recording_enabled=false for session %s — skipping",
                call_session_id,
            )
            return

        # Skip if already uploaded
        if session.recording_s3_path:
            logger.debug(
                "Recording upload: session %s already has recording_s3_path — skipping",
                call_session_id,
            )
            return

        # Retrieve egress_id stored during recording start
        recording_meta = _get_recording_meta(session)
        egress_id = recording_meta.get("egress_id") if recording_meta else None
        gcs_path = recording_meta.get("gcs_path") if recording_meta else None

        if not egress_id or not gcs_path:
            logger.info(
                "Recording upload: no egress_id/gcs_path for session %s — skipping",
                call_session_id,
            )
            return

        # Check LiveKit egress completed
        _check_and_finalize(db, session, egress_id, gcs_path)

    except Exception as exc:
        logger.error(
            "Recording upload sync error for session %s: %s",
            call_session_id,
            exc,
            exc_info=True,
        )
    finally:
        db.close()


def _get_recording_meta(session) -> Optional[dict]:
    """Extract the recording sub-dict from call_session.call_metadata."""
    meta = session.call_metadata
    if not isinstance(meta, dict):
        return None
    return meta.get("recording")


def _check_and_finalize(db, session, egress_id: str, gcs_path: str) -> None:
    """
    Poll LiveKit egress status synchronously (blocking) and finalize the recording.

    LiveKit egress should be COMPLETE shortly after call end.  We do a single
    status check; if FAILED/ABORTED, record the error.  If still in progress,
    we log a warning — no retry in Sprint 2.
    """
    from app.services import s3_recording_service

    # We need an async loop to call LiveKit's async API from sync context.
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(_stop_egress_async(egress_id))
        loop.run_until_complete(asyncio.sleep(1.5))
        egress_info = loop.run_until_complete(_fetch_egress_info_async(egress_id))
    except Exception as exc:
        logger.error("Failed to fetch LiveKit egress info for %s: %s", egress_id, exc)
        _mark_recording_error(db, session)
        return
    finally:
        try:
            loop.close()
        except Exception:
            pass

    if egress_info is None:
        logger.warning("LiveKit egress %s not found — marking recording_error", egress_id)
        _mark_recording_error(db, session)
        return

    # Use integer constants to avoid importing livekit.api at the top of this file.
    # EGRESS_COMPLETE=3, EGRESS_FAILED=4, EGRESS_ABORTED=5
    # These are stable protobuf enum values from livekit-protocol.
    _EGRESS_COMPLETE = 3
    _EGRESS_FAILED = 4
    _EGRESS_ABORTED = 5

    status = egress_info.status

    if status in (_EGRESS_FAILED, _EGRESS_ABORTED):
        logger.error(
            "LiveKit egress %s failed/aborted (status=%s) for session %s",
            egress_id,
            status,
            session.id,
        )
        _mark_recording_error(db, session)
        return

    if status != _EGRESS_COMPLETE:
        logger.warning(
            "LiveKit egress %s status=%s for session %s — not yet complete, no retry in Sprint 2",
            egress_id,
            status,
            session.id,
        )
        _mark_recording_error(db, session)
        return

    # Egress COMPLETE — update S3 metadata then mark DB
    try:
        metadata = {
            "callId": str(session.id),
            "workspaceId": str(session.tenant_id),
            "agentId": str(session.agent_id),
            "duration": str(session.duration or ""),
        }
        s3_recording_service.update_object_metadata(gcs_path, metadata)
    except Exception as exc:
        logger.warning(
            "S3 metadata update failed for %s: %s (continuing — path will still be set)",
            gcs_path,
            exc,
        )

    # Update DB record
    try:
        session.recording_s3_path = gcs_path
        session.recording_error = False
        db.commit()
        logger.info(
            "Recording finalized: session=%s s3_path=%s",
            session.id,
            gcs_path,
        )
    except Exception as exc:
        logger.error(
            "DB update failed for recording session %s: %s",
            session.id,
            exc,
            exc_info=True,
        )
        db.rollback()


async def _fetch_egress_info_async(egress_id: str):
    """Async helper to fetch LiveKit egress info."""
    from app.services.livekit_recording_service import livekit_recording_service

    return await livekit_recording_service.get_egress_info(egress_id)


async def _stop_egress_async(egress_id: str) -> None:
    """Ensure egress is stopped before polling completion status."""
    from app.services.livekit_recording_service import livekit_recording_service

    await livekit_recording_service.stop_room_recording(egress_id)


def _mark_recording_error(db, session) -> None:
    """Set recording_error=True on the call_session.  No retry in Sprint 2."""
    try:
        session.recording_error = True
        db.commit()
    except Exception as exc:
        logger.error("Failed to set recording_error for session %s: %s", session.id, exc)
        db.rollback()
