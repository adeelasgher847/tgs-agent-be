"""
Call Recording Upload Service — post-call async S3 finalization job.

Flow (ARQ-based retry-with-backoff, replacing the old one-shot in-process
check):
  1. schedule_recording_upload() enqueues the ARQ job "finalize_call_recording"
     (attempt=1) with a short initial defer so LiveKit has a moment to begin
     finalizing the egress after stop_egress is requested.
  2. _finalize_call_recording_arq_task() (the ARQ job body) loads the call
     session, resolves the egress_id/gcs_path, requests stop_egress on the
     first attempt only, and polls LiveKit's egress status.
  3. If the egress is COMPLETE: update S3 object metadata (callId,
     workspaceId, agentId, duration), then set recording_s3_path and clear
     recording_error.
  4. If the egress is FAILED/ABORTED: this is a genuine terminal failure —
     mark recording_error=True immediately, no retry.
  5. If the egress is still in-progress (STARTING/ACTIVE/ENDING/etc.), or the
     status fetch was inconclusive (not found / raised an exception): retry by
     re-enqueuing the same job with attempt+1 after a fixed backoff delay, up
     to _MAX_FINALIZE_ATTEMPTS. Once attempts are exhausted (or no ARQ pool is
     available to reschedule), mark recording_error=True.
"""

from __future__ import annotations

import asyncio
import uuid

from app.core.logger import logger

# Total retry window: _MAX_FINALIZE_ATTEMPTS * _FINALIZE_RETRY_DELAY_SECONDS
# (~30s) — generous for LiveKit's typical finalize + S3-upload time on a
# normal-length call recording.
_MAX_FINALIZE_ATTEMPTS = 10
_FINALIZE_RETRY_DELAY_SECONDS = 3

# Short initial delay before the very first status check — stop_egress was
# just requested, give LiveKit a moment before polling.
_INITIAL_FINALIZE_DELAY_SECONDS = 2


def schedule_recording_upload(call_session_id: uuid.UUID) -> None:
    """
    Fire-and-forget: schedule recording finalization after call ends.

    Mirrors schedule_call_summary_generation() in voice_analysis_service.
    Called synchronously (without await) from bidirectional_stream's
    shutdown path, voice.handle_call_events_webhook (status=completed), and
    livekit_browser_call_handler's shutdown path.
    """
    from app.utils.arq_pool import get_arq_pool

    pool = get_arq_pool()
    if pool is None:
        logger.warning(
            "ARQ pool not ready; recording finalization skipped for session=%s",
            call_session_id,
        )
        return

    async def _enqueue() -> None:
        try:
            # _job_id dedups concurrent triggers for the same call (e.g. a
            # Twilio call's AI-side call-ending logic and Twilio's own
            # StatusCallback webhook can both call schedule_recording_upload
            # for the same session) — mirrors execute_callback's use of
            # _job_id for the identical dual-trigger shape.
            await pool.enqueue_job(
                "finalize_call_recording",
                str(call_session_id),
                1,
                _defer_by=_INITIAL_FINALIZE_DELAY_SECONDS,
                _job_id=f"finalize_recording:{call_session_id}:1",
            )
        except Exception as exc:
            logger.warning("Failed to enqueue recording finalization job: %s", exc)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_enqueue())
        return
    asyncio.create_task(_enqueue())


async def _finalize_call_recording_arq_task(
    ctx: dict, call_session_id: str, attempt: int
) -> None:
    """
    ARQ job entrypoint — registered as ``finalize_call_recording`` in
    app/workers/batch_call_worker.py::WorkerSettings.functions.

    Polls LiveKit egress status and finalizes the recording, retrying with
    backoff while the egress is still in-progress or the status fetch is
    inconclusive. See module docstring for the full flow.
    """
    from app.db.session import SessionLocal
    from app.models.call_session import CallSession
    from app.services import s3_recording_service
    from app.services.livekit_recording_service import livekit_recording_service
    from app.services.recording_config_service import get_recording_enabled_for_call

    db = SessionLocal()
    try:
        session_uuid = uuid.UUID(call_session_id)
        session: CallSession | None = db.get(CallSession, session_uuid)
        if session is None:
            logger.warning(
                "Recording finalize: call_session %s not found", call_session_id
            )
            return

        # Skip if recording was not enabled for this call's number (config may
        # have changed mid-flight).
        if not get_recording_enabled_for_call(db, session):
            logger.debug(
                "Recording finalize: recording_enabled=false for session %s — skipping",
                call_session_id,
            )
            return

        # Idempotent — avoid duplicate finalize work if somehow re-triggered.
        if session.recording_s3_path:
            logger.debug(
                "Recording finalize: session %s already has recording_s3_path — skipping",
                call_session_id,
            )
            return

        recording_meta = _get_recording_meta(session)
        egress_id = recording_meta.get("egress_id") if recording_meta else None
        gcs_path = recording_meta.get("gcs_path") if recording_meta else None

        if not egress_id or not gcs_path:
            logger.info(
                "Recording finalize: no egress_id/gcs_path for session %s — skipping",
                call_session_id,
            )
            return

        if attempt == 1:
            # Only request stop on the first attempt — it was already
            # requested once, no need to repeat on retries.
            await livekit_recording_service.stop_room_recording(egress_id)

        try:
            egress_info = await livekit_recording_service.get_egress_info(egress_id)
        except Exception as exc:
            # Transient API/network blips fetching status should not
            # permanently poison the call's recording after only one
            # attempt — treat the same as an inconclusive (None) result.
            logger.warning(
                "Recording finalize: error fetching egress info for %s (attempt=%s): %s",
                egress_id,
                attempt,
                exc,
            )
            egress_info = None

        # Use integer constants to avoid importing livekit.api at the top of
        # this file. EGRESS_COMPLETE=3, EGRESS_FAILED=4, EGRESS_ABORTED=5 —
        # stable protobuf enum values from livekit-protocol.
        _EGRESS_COMPLETE = 3
        _EGRESS_FAILED = 4
        _EGRESS_ABORTED = 5

        status = egress_info.status if egress_info is not None else None

        if status in (_EGRESS_FAILED, _EGRESS_ABORTED):
            logger.error(
                "LiveKit egress %s failed/aborted (status=%s) for session %s",
                egress_id,
                status,
                session.id,
            )
            mark_recording_error(db, session)
            return

        if status == _EGRESS_COMPLETE:
            final_s3_path = _resolve_final_recording_path(
                egress_info, gcs_path, egress_id, session.id
            )

            try:
                metadata = {
                    "callId": str(session.id),
                    "workspaceId": str(session.tenant_id),
                    "agentId": str(session.agent_id),
                    "duration": str(session.duration or ""),
                }
                s3_recording_service.update_object_metadata(final_s3_path, metadata)
            except Exception as exc:
                logger.warning(
                    "S3 metadata update failed for %s: %s (continuing — path will still be set)",
                    final_s3_path,
                    exc,
                )

            try:
                session.recording_s3_path = final_s3_path
                session.recording_error = False
                db.commit()
                logger.info(
                    "Recording finalized: session=%s s3_path=%s",
                    session.id,
                    final_s3_path,
                )
            except Exception as exc:
                logger.error(
                    "DB update failed for recording session %s: %s",
                    session.id,
                    exc,
                    exc_info=True,
                )
                db.rollback()
            return

        # Inconclusive / not yet complete (egress_info is None, fetch raised,
        # or status is STARTING/ACTIVE/ENDING/LIMIT_REACHED/other non-terminal
        # value) — this is the core fix: retry with backoff instead of
        # treating "still finalizing" as a hard failure.
        if attempt < _MAX_FINALIZE_ATTEMPTS:
            logger.info(
                "Recording finalize: egress %s status=%s for session %s (attempt=%s) — "
                "not yet complete, retrying",
                egress_id,
                status,
                session.id,
                attempt,
            )
            pool = ctx.get("redis")
            if pool is not None:
                try:
                    await pool.enqueue_job(
                        "finalize_call_recording",
                        call_session_id,
                        attempt + 1,
                        _defer_by=_FINALIZE_RETRY_DELAY_SECONDS,
                        _job_id=f"finalize_recording:{call_session_id}:{attempt + 1}",
                    )
                    return
                except Exception as exc:
                    logger.warning(
                        "Recording finalize: failed to enqueue retry (attempt=%s) for "
                        "session %s: %s — marking recording_error",
                        attempt,
                        session.id,
                        exc,
                    )
                    mark_recording_error(db, session)
                    return
            # No way to reschedule — fall through to marking the error.
            logger.warning(
                "Recording finalize: no ARQ pool available to reschedule retry for "
                "session %s — marking recording_error",
                session.id,
            )
            mark_recording_error(db, session)
            return

        logger.warning(
            "Recording finalize: retries exhausted (attempt=%s) for session %s "
            "(egress %s status=%s) — marking recording_error",
            attempt,
            session.id,
            egress_id,
            status,
        )
        mark_recording_error(db, session)

    except Exception as exc:
        logger.error(
            "Recording finalize task error for session %s: %s",
            call_session_id,
            exc,
            exc_info=True,
        )
    finally:
        db.close()


def _resolve_final_recording_path(egress_info, gcs_path: str, egress_id: str, session_id) -> str:
    """
    Determine the S3 key to persist as recording_s3_path once egress is COMPLETE.

    Our own code pre-computes ``gcs_path`` (via s3_recording_service.build_s3_key())
    *before* egress ever runs and sends it as the requested ``filepath`` in the
    EncodedFileOutput. LiveKit's egress service does not use that filepath
    verbatim: it appends its own container extension server-side when the
    requested path doesn't already end with one it recognizes (see
    livekit/egress's FileConfig.updateFilepath), so the object actually written
    to S3 can differ from what we requested (e.g. our ".opus" request lands as
    ".opus.ogg" in S3).

    ``EgressInfo.file_results[0].filename`` is confirmed (via both static
    protobuf introspection and a live completed egress) to be the *actual*,
    bucket-relative S3 key LiveKit wrote to — the same format build_s3_key()
    produces and generate_signed_url()/get_object_size() expect.
    ``file_results[0].location`` is a full ``https://bucket.s3.../key`` URL,
    not a bucket-relative key, so it's deliberately not used here.

    The current single-EncodedFileOutput egress request (livekit_recording_service
    .start_room_recording) structurally produces at most one meaningful
    file_results entry, but this still defensively checks for an empty/missing
    list rather than assuming index 0 exists.
    """
    file_results = getattr(egress_info, "file_results", None) if egress_info is not None else None

    if not file_results:
        logger.info(
            "Recording finalize: egress %s returned no file_results for session %s — "
            "falling back to pre-computed path %s",
            egress_id,
            session_id,
            gcs_path,
        )
        return gcs_path

    filename = getattr(file_results[0], "filename", None)
    if isinstance(filename, str) and filename.strip():
        if filename != gcs_path:
            logger.info(
                "Recording finalize: using LiveKit-confirmed S3 path for session %s "
                "(egress=%s): %s (pre-computed path was: %s)",
                session_id,
                egress_id,
                filename,
                gcs_path,
            )
        return filename

    logger.warning(
        "Recording finalize: egress %s file_results[0].filename unusable (%r) "
        "for session %s — falling back to pre-computed path %s",
        egress_id,
        filename,
        session_id,
        gcs_path,
    )
    return gcs_path


def _get_recording_meta(session) -> dict | None:
    """Extract the recording sub-dict from call_session.call_metadata."""
    meta = session.call_metadata
    if not isinstance(meta, dict):
        return None
    return meta.get("recording")


def mark_recording_error(db, session) -> None:
    """Best-effort: set recording_error=True on the call_session so
    GET /api/v1/recordings/{id} can report a genuine failure instead of the
    generic "still processing" 404. Never raises. Shared by both the
    Twilio finalize-job path (this module) and the browser-call start path
    (livekit_browser_call_handler._start_browser_call_recording)."""
    try:
        session.recording_error = True
        db.commit()
    except Exception as exc:
        logger.error("Failed to set recording_error for session %s: %s", session.id, exc)
        db.rollback()
