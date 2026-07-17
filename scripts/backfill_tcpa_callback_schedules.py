"""
Backfill script: reschedule existing pending CallbackSchedule rows with a
future scheduled_at that would fall outside the TCPA-compliant calling
window (agent's flow business_hours, clamped to 8am-9pm local time for US
workspaces) once converted to the agent's callback_timezone.

Also cancels rows belonging to agents with a missing/invalid
callback_timezone, since those cannot be safely dispatched.

Run from project root:
    ./venv/bin/python -m scripts.backfill_tcpa_callback_schedules
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.agent import Agent
from app.models.callback_schedule import CallbackSchedule
from app.services.callback_scheduler_service import (
    InvalidCallbackTimezoneError,
    callback_scheduler_service as svc,
)
from app.core.logger import logger


def main() -> None:
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)

        schedules = (
            db.execute(
                select(CallbackSchedule).where(
                    CallbackSchedule.status == "pending",
                    CallbackSchedule.scheduled_at > now,
                )
            )
            .scalars()
            .all()
        )

        if not schedules:
            print("[INFO] No future pending callback schedules found.")
            return

        print(
            f"[INFO] Found {len(schedules)} future pending callback schedule(s). Checking..."
        )

        rescheduled = 0
        cancelled = 0
        skipped = 0

        for schedule in schedules:
            agent = db.get(Agent, schedule.agent_id)
            if agent is None:
                skipped += 1
                continue

            try:
                tz_name = svc._require_valid_callback_timezone(agent)
            except InvalidCallbackTimezoneError:
                schedule.status = "cancelled"
                db.commit()
                svc._alert_invalid_callback_timezone(db, agent)
                cancelled += 1
                print(
                    f"[CANCEL] schedule_id={schedule.id} agent_id={agent.id} "
                    f"reason=invalid_callback_timezone value={agent.callback_timezone!r}"
                )
                continue

            local_dt = schedule.scheduled_at.astimezone(ZoneInfo(tz_name))
            flow, is_us = svc._resolve_dispatch_context(
                db, agent, schedule.original_call
            )

            if svc._is_within_business_hours(local_dt, flow, is_us):
                skipped += 1
                continue

            original_scheduled_at = schedule.scheduled_at
            next_window = svc._next_valid_window(
                schedule.scheduled_at,
                tz_name,
                flow,
                is_us,
                agent_id=agent.id,
            )
            schedule.scheduled_at = next_window
            db.commit()

            logger.info(
                "callback_rescheduled",
                extra={
                    "action": "callback.rescheduled_for_compliance",
                    "original_scheduled_at": original_scheduled_at.isoformat(),
                    "new_scheduled_at": next_window.isoformat(),
                    "reason": "outside_business_hours_in_timezone",
                },
            )
            rescheduled += 1
            print(
                f"[RESCHEDULE] schedule_id={schedule.id} agent_id={agent.id} "
                f"{original_scheduled_at.isoformat()} -> {next_window.isoformat()}"
            )

        print("\n=== Backfill Summary ===")
        print(f"total_checked={len(schedules)}")
        print(f"rescheduled={rescheduled}")
        print(f"cancelled_invalid_timezone={cancelled}")
        print(f"skipped_already_compliant_or_missing_agent={skipped}")
        print("done")
    finally:
        db.close()


if __name__ == "__main__":
    main()
