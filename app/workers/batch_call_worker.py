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
from datetime import datetime, timezone
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


# ── WorkerSettings ────────────────────────────────────────────────────────────

class WorkerSettings:
    """
    ARQ WorkerSettings consumed by `arq app.workers.batch_call_worker.WorkerSettings`.

    Environment requirements:
      REDIS_URL       — Redis connection string (already used by rate limiter)
      DATABASE_URL    — PostgreSQL sync URL (already in Settings)
      N8N_WEBHOOK_SECRET — used by the fake-request auth bridge in BatchCallWorkerService

    To start the worker:
        arq app.workers.batch_call_worker.WorkerSettings
    """

    functions = [process_batch_job, poll_pending_batch_jobs, retry_webhook_delivery]

    cron_jobs = [
        # Poll every 60 seconds for orphaned pending jobs
        arq_cron(poll_pending_batch_jobs, second={0}, run_at_startup=True),
    ]

    @staticmethod
    def redis_settings():
        import arq  # type: ignore

        return arq.connections.RedisSettings.from_dsn(settings.REDIS_URL)

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
        _arq.cron(poll_pending_batch_jobs, second={0}, run_at_startup=True),
    ]
except ImportError:
    WorkerSettings.cron_jobs = []
