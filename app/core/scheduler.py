"""
APScheduler setup for the Smart Callback Scheduler.

Uses BackgroundScheduler with a SQLAlchemy (PostgreSQL) job store so that
scheduled jobs survive pod restarts.  APScheduler manages its own
`apscheduler_jobs` table in the same database.

Lifecycle:
    start_scheduler()  — called once at application startup (lifespan)
    stop_scheduler()   — called at application shutdown (lifespan)
"""
from __future__ import annotations

from datetime import timezone

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.core.config import settings
from app.core.logger import logger


def _poll_pending_callbacks() -> None:
    """
    APScheduler job function.  Runs every 30 seconds.

    Opens its own synchronous DB session (required because APScheduler runs
    jobs in a thread-pool, outside of any FastAPI request context).
    """
    from app.db.session import SessionLocal
    from app.services.callback_scheduler_service import callback_scheduler_service

    db = SessionLocal()
    try:
        callback_scheduler_service.process_pending_callbacks(db)
    except Exception as exc:  # noqa: BLE001
        logger.error("callback_poll_job error: %s", exc, exc_info=True)
    finally:
        db.close()


_JOB_ID = "smart_callback_poll"


def _build_scheduler() -> BackgroundScheduler:
    jobstores = {
        "default": SQLAlchemyJobStore(url=settings.DATABASE_URL),
    }
    executors = {
        "default": ThreadPoolExecutor(max_workers=4),
    }
    job_defaults = {
        "coalesce": True,   # merge missed runs into one
        "max_instances": 1, # never run more than one poll concurrently
    }
    return BackgroundScheduler(
        jobstores=jobstores,
        executors=executors,
        job_defaults=job_defaults,
        timezone=timezone.utc,
    )


# Module-level singleton — created once, reused across start/stop calls.
_scheduler: BackgroundScheduler = _build_scheduler()


def start_scheduler() -> None:
    """Register the polling job and start the scheduler."""
    if _scheduler.running:
        logger.warning("APScheduler is already running; start_scheduler() called twice")
        return

    # Replace any stale job stored in the DB so config changes take effect.
    _scheduler.add_job(
        _poll_pending_callbacks,
        trigger=IntervalTrigger(seconds=30),
        id=_JOB_ID,
        name="Smart Callback Poller",
        replace_existing=True,
    )

    _scheduler.start()
    logger.info("APScheduler started — polling for pending callbacks every 30 s")


def stop_scheduler() -> None:
    """Gracefully shut down the scheduler on application exit."""
    if _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("APScheduler stopped")
