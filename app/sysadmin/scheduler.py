"""
Nightly APScheduler job — recomputes sys_request_stats for the current month
at midnight UTC so the Overview endpoint reads from pre-aggregated rows.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def recompute_current_month() -> None:
    from app.db.session import SessionLocal
    from app.sysadmin.stats.service import recompute_stats

    month = datetime.now(timezone.utc).strftime("%Y-%m")
    db = SessionLocal()
    try:
        logger.info("SysAdmin nightly recompute starting for %s", month)
        recompute_stats(db, month)
        logger.info("SysAdmin nightly recompute complete for %s", month)
    except Exception:
        logger.exception("SysAdmin nightly recompute failed for %s", month)
    finally:
        db.close()


def register_sysadmin_jobs(scheduler) -> None:
    """Register with the BackgroundScheduler created in app/main.py lifespan."""
    scheduler.add_job(
        recompute_current_month,
        trigger="cron",
        hour=0,
        minute=0,
        second=0,
        id="sysadmin_nightly_recompute",
        replace_existing=True,
        name="SysAdmin: nightly stats recompute",
    )
    logger.info("SysAdmin nightly recompute job registered")
