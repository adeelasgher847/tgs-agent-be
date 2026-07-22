"""
ARQ batch-call worker.

Run locally:
    arq app.workers.batch_call_worker.WorkerSettings

In Docker / Kubernetes, add a second container/process with the same image:
    CMD ["arq", "app.workers.batch_call_worker.WorkerSettings"]

The worker:
  1. Picks up to MAX_BATCH_CONCURRENCY waiting records per job (SKIP LOCKED).
  2. Dispatches each via the existing initiate_call path.
  3. Respects OUTBOUND_MAX_CONCURRENT_PER_WORKSPACE (checked inside initiate_call).
  4. Polls every BATCH_WORKER_POLL_INTERVAL_SEC for scheduled/pending jobs.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.core.config import settings
from app.core.logger import logger

# ── Job functions ─────────────────────────────────────────────────────────────


async def process_batch_job(ctx: dict, batch_job_id: str) -> None:
    """
    ARQ job function — processes one batch_job by picking and dispatching calls.

    Picks MAX_BATCH_CONCURRENCY records at a time until the job is drained or
    the workspace concurrent-call ceiling is hit.
    """
    from app.db.session import SessionLocal
    from app.models.agent import Agent
    from app.models.batch_job import BatchJob
    from app.services.batch_call_worker_service import BatchCallWorkerService

    job_id = uuid.UUID(batch_job_id)
    db = SessionLocal()
    try:
        job = db.get(BatchJob, job_id)
        if job is None:
            logger.warning("process_batch_job: BatchJob %s not found", batch_job_id)
            return

        if job.status in ("cancelled", "completed", "failed"):
            logger.info("process_batch_job: BatchJob %s is %s — skipping", batch_job_id, job.status)
            return

        # Respect scheduled_at — do not start early
        if job.scheduled_at:
            now = datetime.now(timezone.utc)
            scheduled = job.scheduled_at
            if scheduled.tzinfo is None:
                scheduled = scheduled.replace(tzinfo=timezone.utc)
            if now < scheduled:
                logger.info(
                    "process_batch_job: BatchJob %s not yet due (scheduled_at=%s)",
                    batch_job_id,
                    scheduled,
                )
                return

        # Load agent once for prompt-variable substitution
        agent = db.get(Agent, job.agent_id)
        agent_system_prompt: Optional[str] = agent.system_prompt if agent else None

        svc = BatchCallWorkerService(db)
        batch_limit = settings.MAX_BATCH_CONCURRENCY

        records = svc.pick_waiting_records(job_id, batch_limit)
        if not records:
            logger.info(
                "process_batch_job: no waiting records for BatchJob %s", batch_job_id
            )
            return

        logger.info(
            "process_batch_job: dispatching %d records for BatchJob %s",
            len(records),
            batch_job_id,
        )

        # Dispatch concurrently within the picked batch
        await asyncio.gather(
            *[
                svc.dispatch_record(
                    record=rec,
                    workspace_id=job.workspace_id,
                    agent_id=job.agent_id,
                    agent_system_prompt=agent_system_prompt,
                )
                for rec in records
            ],
            return_exceptions=True,
        )

        # Re-enqueue self if more waiting records remain
        job = db.get(BatchJob, job_id)  # refresh
        if job and (job.waiting_count or 0) > 0 and job.status not in ("cancelled", "completed"):
            logger.info(
                "process_batch_job: %d records still waiting in BatchJob %s — re-enqueuing",
                job.waiting_count,
                batch_job_id,
            )
            # Re-enqueue so the next batch fires shortly after
            if ctx.get("redis"):
                await ctx["redis"].enqueue_job("process_batch_job", batch_job_id)

    finally:
        db.close()


async def retry_webhook_delivery(
    ctx: dict, delivery_id: str, attempt_number: int
) -> None:
    """
    ARQ job function — re-attempts a failed webhook delivery.

    delivery_id: str UUID of the WebhookDelivery row.
    attempt_number: 1-indexed retry count (1 = first retry after initial failure).
    """
    import uuid as _uuid

    from app.services.webhook_service import retry_webhook_delivery as _retry

    await _retry(
        delivery_id=_uuid.UUID(delivery_id),
        attempt_number=attempt_number,
    )


async def poll_pending_batch_jobs(ctx: dict) -> None:
    """
    Periodic cron job — picks up any pending/processing jobs that have
    no in-flight ARQ task (e.g. after a worker restart).

    Fires every BATCH_WORKER_POLL_INTERVAL_SEC seconds.
    """
    from sqlalchemy import select

    from app.db.session import SessionLocal
    from app.models.batch_job import BatchJob

    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        jobs = (
            db.execute(
                select(BatchJob)
                .where(
                    BatchJob.status.in_(["pending", "processing"]),
                    # Only pick up jobs that are due
                    (BatchJob.scheduled_at.is_(None)) | (BatchJob.scheduled_at <= now),
                )
                .limit(50)
            )
            .scalars()
            .all()
        )

        for job in jobs:
            if job.waiting_count and job.waiting_count > 0:
                logger.info(
                    "poll_pending_batch_jobs: re-enqueuing BatchJob %s (%d waiting)",
                    job.id,
                    job.waiting_count,
                )
                if ctx.get("redis"):
                    await ctx["redis"].enqueue_job("process_batch_job", str(job.id))
    finally:
        db.close()


# ── KB ingestion task ─────────────────────────────────────────────────────────


async def kb_ingestion_task(ctx: dict, file_id: str) -> None:
    """
    ARQ job: download the KB file from S3, extract text, chunk via tiktoken,
    embed with OpenAI ada-002, insert kb_chunks, and update kb_file.status.
    """
    from app.db.session import SessionLocal
    from app.models.kb_file import KbFile
    from app.services.kb_ingestion_service import run_file_ingestion

    fid = uuid.UUID(file_id)
    db = SessionLocal()
    try:
        kb_file = db.get(KbFile, fid)
        if kb_file is None:
            logger.warning("kb_ingestion_task: KbFile %s not found", file_id)
            return

        if kb_file.status != "processing":
            logger.info("kb_ingestion_task: KbFile %s already %s — skipping", file_id, kb_file.status)
            return

        # Download bytes from S3
        file_bytes: Optional[bytes] = None
        if kb_file.s3_path and settings.S3_KB_BUCKET:
            try:
                from app.services.s3_service import get_s3_client

                s3_client = get_s3_client()
                response = s3_client.get_object(
                    Bucket=settings.S3_KB_BUCKET,
                    Key=kb_file.s3_path,
                )
                file_bytes = response["Body"].read()
            except Exception as e:
                logger.error(
                    "kb_ingestion_task: S3 download failed for file_id=%s: %s", file_id, e, exc_info=True
                )
                kb_file.status = "error"
                kb_file.error_message = f"S3 download failed: {str(e)[:500]}"
                db.commit()
                return

        if file_bytes is None:
            kb_file.status = "error"
            kb_file.error_message = "No S3 path set or S3_KB_BUCKET not configured"
            db.commit()
            return

        api_key = settings.OPENAI_API_KEY
        if not api_key:
            kb_file.status = "error"
            kb_file.error_message = "OPENAI_API_KEY not configured"
            db.commit()
            return

        try:
            chunk_count = await run_file_ingestion(
                db=db,
                file_id=fid,
                file_bytes=file_bytes,
                file_type=kb_file.file_type or "",
                api_key=api_key,
            )
            kb_file.status = "ready"
            kb_file.chunk_count = chunk_count
            db.commit()
            logger.info(
                "kb_ingestion_task: file_id=%s ingested %d chunks", file_id, chunk_count
            )
        except Exception as e:
            db.rollback()
            logger.error(
                "kb_ingestion_task: ingestion failed for file_id=%s: %s", file_id, e, exc_info=True
            )
            kb_file = db.get(KbFile, fid)
            if kb_file:
                kb_file.status = "error"
                kb_file.error_message = str(e)[:1000]
                db.commit()
    finally:
        db.close()


# ── GDPR data export task ───────────────────────────────────────────────────


async def run_data_export_job(ctx: dict, export_job_id: str) -> None:
    """
    ARQ job: build the workspace data-export ZIP, upload it to GCS, and
    email the signed download URL to the workspace admin.
    """
    from app.db.session import SessionLocal
    from app.models.data_export_job import DataExportJob
    from app.services.data_export_service import run_export_job

    jid = uuid.UUID(export_job_id)
    db = SessionLocal()
    try:
        job = db.get(DataExportJob, jid)
        if job is None:
            logger.warning("run_data_export_job: DataExportJob %s not found", export_job_id)
            return
        run_export_job(db, job)
    finally:
        db.close()


# ── GoHighLevel (GHL) post-call write-back ────────────────────────────────────


async def ghl_post_call_writeback(ctx: dict, call_session_id: str) -> None:
    """
    ARQ job: create a GHL note summarizing a completed call. Enqueued by
    app.services.ghl_service.schedule_ghl_writeback on call completion.
    """
    from app.services.ghl_service import _post_call_writeback_arq_task

    await _post_call_writeback_arq_task(ctx, call_session_id)


# ── Smart Callback tasks (ARQ-based replacement for APScheduler polling) ─────


async def execute_callback(ctx: dict, schedule_id: str) -> None:
    """
    ARQ job: dispatch a single pending CallbackSchedule and chain the next
    attempt (or exhaust the sequence).

    Idempotency: the row is locked with SELECT ... FOR UPDATE SKIP LOCKED.
    A second worker that picks up the same job_id sees the row locked (or
    already executed) and exits early.

    _job_id is set to ``callback:<schedule_id>`` by the enqueuer so ARQ
    deduplicates concurrent enqueue calls for the same schedule.
    """
    from sqlalchemy import select

    from app.db.session import SessionLocal
    from app.models.callback_schedule import CallbackSchedule
    from app.services.callback_scheduler_service import callback_scheduler_service

    sid = uuid.UUID(schedule_id)
    db = SessionLocal()
    try:
        # Advisory lock prevents two workers from dispatching the same row.
        schedule = db.execute(
            select(CallbackSchedule)
            .where(
                CallbackSchedule.id == sid,
                CallbackSchedule.status == "pending",
            )
            .with_for_update(skip_locked=True)
        ).scalar_one_or_none()

        if schedule is None:
            logger.info(
                "execute_callback: schedule %s not pending or locked — skipping",
                schedule_id,
            )
            return

        pool = ctx.get("redis")
        next_schedule = await callback_scheduler_service.dispatch_and_advance_async(
            db, schedule
        )

        # Enqueue the next ARQ job (either a rescheduled same row or a new attempt).
        if next_schedule is not None and pool is not None:
            job = await pool.enqueue_job(
                "execute_callback",
                str(next_schedule.id),
                _defer_until=next_schedule.scheduled_at,
                _job_id=f"callback:{next_schedule.id}",
            )
            if job is not None:
                next_schedule.arq_job_id = job.job_id
                db.commit()

    except Exception as exc:
        logger.error(
            "execute_callback error schedule=%s: %s", schedule_id, exc, exc_info=True
        )
        db.rollback()
    finally:
        db.close()


async def poll_pending_callbacks(ctx: dict) -> None:
    """
    ARQ cron: recovery job that runs every 60 s.

    Finds ``pending`` CallbackSchedule rows that have no ARQ job ID (i.e. newly
    created by ``maybe_schedule_callback`` or cleared after a business-hours
    reschedule) and submits a deferred ``execute_callback`` job for each.

    This is the only path that bootstraps the ARQ chain after a callback row is
    first written.  ``execute_callback`` itself handles all subsequent chaining.

    ARQ deduplicates on ``_job_id=callback:<schedule_id>`` so re-queuing an
    already-pending job is a safe no-op.
    """
    from sqlalchemy import select

    from app.db.session import SessionLocal
    from app.models.callback_schedule import CallbackSchedule

    pool = ctx.get("redis")
    if pool is None:
        logger.warning("poll_pending_callbacks: no redis pool in ctx — skipping")
        return

    db = SessionLocal()
    try:
        rows = (
            db.execute(
                select(CallbackSchedule)
                .where(
                    CallbackSchedule.status == "pending",
                    CallbackSchedule.arq_job_id.is_(None),
                )
                .limit(100)
            )
            .scalars()
            .all()
        )

        if not rows:
            return

        enqueued = 0
        for schedule in rows:
            job = await pool.enqueue_job(
                "execute_callback",
                str(schedule.id),
                _defer_until=schedule.scheduled_at,
                _job_id=f"callback:{schedule.id}",
            )
            if job is not None:
                schedule.arq_job_id = job.job_id
                enqueued += 1

        db.commit()
        logger.info("poll_pending_callbacks: enqueued %d callback jobs", enqueued)

    except Exception as exc:
        logger.error("poll_pending_callbacks error: %s", exc, exc_info=True)
        db.rollback()
    finally:
        db.close()


async def startup_recover_callbacks(ctx: dict) -> None:
    """
    ARQ on_startup hook — runs exactly once each time a worker process starts.

    Finds every ``pending`` CallbackSchedule row in PostgreSQL and submits an
    ARQ deferred job for it.  This is the sole recovery mechanism for callbacks
    that could not be enqueued at call-end time (Redis unavailable, app crash,
    or process restart while jobs were in-flight).

    Safety: ``_job_id=callback:<schedule_id>`` is an ARQ idempotency key.
    If the job is already sitting in Redis (normal restart, no data loss),
    ARQ returns the existing job without creating a duplicate.  No callback
    can fire twice from this path.

    This function is NOT periodic — it runs once per worker startup and then
    exits.  The event-driven ``_fire_callback_enqueue`` path (called from
    ``update_call_session_status``) handles all new callbacks in real time.
    """
    from sqlalchemy import select

    from app.db.session import SessionLocal
    from app.models.callback_schedule import CallbackSchedule

    pool = ctx.get("redis")
    if pool is None:
        logger.warning("startup_recover_callbacks: no redis pool in ctx — skipping")
        return

    db = SessionLocal()
    try:
        rows = (
            db.execute(
                select(CallbackSchedule).where(CallbackSchedule.status == "pending")
            )
            .scalars()
            .all()
        )

        if not rows:
            logger.info("startup_recover_callbacks: no pending callbacks to recover")
            return

        recovered = 0
        for schedule in rows:
            job = await pool.enqueue_job(
                "execute_callback",
                str(schedule.id),
                _defer_until=schedule.scheduled_at,
                _job_id=f"callback:{schedule.id}",
            )
            if job is not None:
                recovered += 1

        logger.info(
            "startup_recover_callbacks: submitted %d pending callback job(s)", recovered
        )

    except Exception as exc:
        logger.error("startup_recover_callbacks error: %s", exc, exc_info=True)
    finally:
        db.close()


# ── Outbound number reputation monitoring ───────────────────────────────────

async def check_all_phone_numbers_reputation(ctx: dict) -> None:
    """
    Daily cron: refresh the reputation record for every active phone number
    whose last check is missing or older than 24 hours.
    """
    from sqlalchemy import or_, select

    from app.db.session import SessionLocal
    from app.models.phone_number import PhoneNumber
    from app.models.phone_number_reputation import PhoneNumberReputation
    from app.services.reputation_service import check_number_reputation

    db = SessionLocal()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

        rows = db.execute(
            select(PhoneNumber)
            .outerjoin(
                PhoneNumberReputation,
                PhoneNumberReputation.phone_number_id == PhoneNumber.id,
            )
            .where(
                PhoneNumber.status == "active",
                or_(
                    PhoneNumberReputation.last_checked_at.is_(None),
                    PhoneNumberReputation.last_checked_at < cutoff,
                ),
            )
        ).scalars().all()

        logger.info("check_all_phone_numbers_reputation: %d number(s) due for a check", len(rows))

        checked = 0
        for phone_number_obj in rows:
            try:
                await check_number_reputation(db, phone_number_obj)
                checked += 1
            except Exception as exc:
                logger.error(
                    "check_all_phone_numbers_reputation: check failed for %s: %s",
                    phone_number_obj.id,
                    exc,
                    exc_info=True,
                )
                db.rollback()

        logger.info("check_all_phone_numbers_reputation: checked %d/%d number(s)", checked, len(rows))
    finally:
        db.close()


# ── WorkerSettings ────────────────────────────────────────────────────────────

async def purge_old_audit_logs(ctx: dict) -> None:
    """
    Daily cron: delete auditlog rows older than 90 days.

    Sets the session GUC app.bypass_audit_delete = 'true' so the no_delete_audit
    trigger allows the DELETE to proceed (application-code deletes remain blocked).
    """
    from sqlalchemy import text

    from app.db.session import SessionLocal

    try:
        with SessionLocal() as db:
            db.execute(text("SET LOCAL app.bypass_audit_delete = 'true'"))
            result = db.execute(
                text(
                    "DELETE FROM auditlog "
                    "WHERE timestamp < NOW() - INTERVAL '90 days'"
                )
            )
            db.commit()
            logger.info("audit_log retention: deleted %d rows older than 90 days", result.rowcount)
    except Exception as exc:
        logger.error("audit_log retention job failed: %s", exc, exc_info=True)


class WorkerSettings:
    """
    ARQ WorkerSettings consumed by `arq app.workers.batch_call_worker.WorkerSettings`.

    Environment requirements:
      REDIS_URL       — Redis connection string (already used by rate limiter)
      DATABASE_URL    — PostgreSQL sync URL (already in Settings)

    To start the worker:
        arq app.workers.batch_call_worker.WorkerSettings
    """

    # startup hook: runs once per worker process start, never periodically
    on_startup = startup_recover_callbacks

    functions = [
        process_batch_job,
        poll_pending_batch_jobs,
        retry_webhook_delivery,
        kb_ingestion_task,
        execute_callback,
        ghl_post_call_writeback,
        purge_old_audit_logs,
        run_data_export_job,
        # poll_pending_callbacks kept for manual/admin invocation; not in cron_jobs
        poll_pending_callbacks,
        check_all_phone_numbers_reputation,
    ]

    # Replaced at module-load time by the try/except block below
    # (arq_cron is not yet defined at class-body evaluation time).
    cron_jobs = []

    # arq's `get_kwargs()` reads WorkerSettings.__dict__ directly — bypassing
    # the descriptor protocol entirely — so redis_settings must be a plain
    # value here, not a @staticmethod/@property (those show up in __dict__ as
    # the raw descriptor object, not its return value, and crash create_pool
    # with "'staticmethod' object has no attribute 'host'"). Verified by
    # actually running the worker against this exact pattern.
    import arq as _arq_module  # type: ignore

    redis_settings = _arq_module.connections.RedisSettings.from_dsn(settings.REDIS_URL)
    del _arq_module

    # Default is 3600s; lowered so `arq ... --check` (used by the Docker Compose
    # healthcheck) reflects liveness within the container's start_period.
    health_check_interval = 30

    max_jobs = 50
    job_timeout = 300  # seconds


# Deferred import so the file can be imported without arq installed
def arq_cron(*args, **kwargs):
    import arq  # type: ignore

    return arq.cron(*args, **kwargs)


# Fix: replace placeholder with real arq.cron at import time
try:
    import arq as _arq  # type: ignore

    WorkerSettings.cron_jobs = [
        # Batch job recovery: re-enqueue orphaned pending batch jobs every 60 s
        _arq.cron(poll_pending_batch_jobs, second={0}, run_at_startup=True),
        # Audit log 90-day retention: runs once per day at 03:00 UTC
        _arq.cron(purge_old_audit_logs, hour={3}, minute={0}, second={0}),
        # Outbound number reputation refresh: runs once per day at 02:00 UTC
        _arq.cron(check_all_phone_numbers_reputation, hour={2}, minute={0}, second={0}),
        # Callback recovery is handled by WorkerSettings.on_startup (startup_recover_callbacks),
        # not a periodic cron — no polling loop needed for correctness.
    ]
except ImportError:
    WorkerSettings.cron_jobs = []
