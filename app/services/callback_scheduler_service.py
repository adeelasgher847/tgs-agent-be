from __future__ import annotations

import uuid
from datetime import datetime, timedelta, time
from typing import List, Optional
from zoneinfo import ZoneInfo

from fastapi import HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.core.logger import logger
from app.models.agent import Agent
from app.models.business_hours import BusinessHours
from app.models.callback_schedule import CallbackSchedule
from app.models.call_session import CallSession
from app.schemas.callback_scheduler import (
    CallbackConfigUpdate,
    CallbackConfigResponse,
    CallbackStatusResponse,
    CallbackHistoryItem,
    GapInterval,
)


# ── Constants ──────────────────────────────────────────────────────────────────

CALLBACK_TRIGGER_STATUSES = frozenset({"no_answer", "busy"})
_MAX_BUSINESS_HOURS_LOOKAHEAD_DAYS = 14  # give up after 2 weeks with no open window


# ── Service ────────────────────────────────────────────────────────────────────


class CallbackSchedulerService:
    """
    Encapsulates all Smart Callback Scheduler business logic.

    Responsibilities:
    - Store / retrieve per-agent callback config (smart_callback_enabled,
      max_callback_attempts, callback_gap_schedule, callback_timezone).
    - Create a CallbackSchedule record when a call ends in no_answer / busy.
    - Process pending callbacks: enforce business hours, dispatch the call,
      and chain the next attempt or mark the sequence exhausted.
    """

    # ── Config endpoints ───────────────────────────────────────────────────────

    def update_callback_config(
        self,
        db: Session,
        agent_id: uuid.UUID,
        tenant_id: uuid.UUID,
        payload: CallbackConfigUpdate,
    ) -> CallbackConfigResponse:
        agent = self._get_agent(db, agent_id, tenant_id)

        agent.smart_callback_enabled = payload.smart_callback_enabled
        agent.max_callback_attempts = payload.max_attempts
        agent.callback_gap_schedule = [
            {"days": g.days, "hours": g.hours} for g in payload.gap_schedule
        ]
        agent.callback_timezone = payload.timezone

        db.add(agent)
        db.commit()
        db.refresh(agent)

        logger.info(
            "callback_config_updated agent_id=%s enabled=%s max_attempts=%s",
            agent_id,
            payload.smart_callback_enabled,
            payload.max_attempts,
        )
        return CallbackConfigResponse(
            smart_callback_enabled=agent.smart_callback_enabled,
            max_attempts=agent.max_callback_attempts,
            gap_schedule=[GapInterval(**g) for g in (agent.callback_gap_schedule or [])],
            timezone=agent.callback_timezone,
        )

    def get_callback_status(
        self,
        db: Session,
        agent_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> CallbackStatusResponse:
        agent = self._get_agent(db, agent_id, tenant_id)

        pending_count: int = db.execute(
            select(func.count(CallbackSchedule.id)).where(
                CallbackSchedule.agent_id == agent_id,
                CallbackSchedule.status == "pending",
            )
        ).scalar_one()

        next_scheduled_at: Optional[datetime] = db.execute(
            select(func.min(CallbackSchedule.scheduled_at)).where(
                CallbackSchedule.agent_id == agent_id,
                CallbackSchedule.status == "pending",
            )
        ).scalar_one()

        return CallbackStatusResponse(
            enabled=agent.smart_callback_enabled,
            max_attempts=agent.max_callback_attempts,
            gap_schedule=[GapInterval(**g) for g in (agent.callback_gap_schedule or [])],
            timezone=agent.callback_timezone,
            pending_retries=pending_count,
            next_scheduled_at=next_scheduled_at,
        )

    def get_callback_history(
        self,
        db: Session,
        call_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> List[CallbackHistoryItem]:
        # Verify the call belongs to this tenant before exposing history
        call = db.execute(
            select(CallSession).where(
                CallSession.id == call_id,
                CallSession.tenant_id == tenant_id,
            )
        ).scalar_one_or_none()

        if call is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Call session not found",
            )

        rows = db.execute(
            select(CallbackSchedule)
            .where(CallbackSchedule.original_call_id == call_id)
            .order_by(CallbackSchedule.attempt_number)
        ).scalars().all()

        return [
            CallbackHistoryItem(
                attempt_number=r.attempt_number,
                scheduled_at=r.scheduled_at,
                executed_at=r.executed_at,
                status=r.status,
            )
            for r in rows
        ]

    # ── Trigger logic (called from call-status webhook / call-end handler) ─────

    def maybe_schedule_callback(
        self,
        db: Session,
        call_session: CallSession,
    ) -> Optional[CallbackSchedule]:
        """
        Called when a CallSession's status is updated.
        Creates the first CallbackSchedule record if the agent has smart
        callbacks enabled and the call ended as no_answer or busy.

        Returns the created CallbackSchedule or None.
        """
        if call_session.status not in CALLBACK_TRIGGER_STATUSES:
            return None

        agent: Optional[Agent] = db.get(Agent, call_session.agent_id)
        if agent is None or not agent.smart_callback_enabled:
            return None

        gap_schedule: List[dict] = agent.callback_gap_schedule or []
        if not gap_schedule:
            logger.warning(
                "smart_callback_enabled but gap_schedule is empty for agent %s",
                agent.id,
            )
            return None

        # The first callback uses gap_schedule[0]
        first_gap = gap_schedule[0]
        base_time = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
        target_at = base_time + timedelta(
            days=first_gap.get("days", 0),
            hours=first_gap.get("hours", 0),
        )

        # Enforce business hours for the first attempt
        tz_name = agent.callback_timezone or "UTC"
        scheduled_at = self._next_valid_window(db, agent, target_at, tz_name)

        phone = call_session.customer_phone_number or call_session.to_number
        if not phone:
            logger.warning(
                "Cannot schedule callback for call %s: no destination phone number",
                call_session.id,
            )
            return None

        schedule = CallbackSchedule(
            original_call_id=call_session.id,
            agent_id=agent.id,
            phone_number=phone,
            attempt_number=1,
            scheduled_at=scheduled_at,
            timezone=tz_name,
            status="pending",
        )
        db.add(schedule)
        db.commit()
        db.refresh(schedule)

        logger.info(
            "callback_scheduled call_id=%s attempt=1 at=%s tz=%s",
            call_session.id,
            scheduled_at.isoformat(),
            tz_name,
        )
        return schedule

    # ── APScheduler job (runs every 30 s) ─────────────────────────────────────

    def process_pending_callbacks(self, db: Session) -> None:
        """
        Poll for due callbacks and dispatch each one.
        Designed to be called by the APScheduler IntervalTrigger job.

        Each iteration selects a single row with FOR UPDATE SKIP LOCKED so the
        lock is scoped to exactly one transaction. Committing (or rolling back)
        inside _dispatch_and_advance releases only that row's lock, which means
        other workers/pods can safely claim any remaining rows without risk of
        double-dispatch.
        """
        now = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))

        while True:
            schedule = db.execute(
                select(CallbackSchedule)
                .where(
                    CallbackSchedule.status == "pending",
                    CallbackSchedule.scheduled_at <= now,
                )
                .order_by(CallbackSchedule.scheduled_at)
                .with_for_update(skip_locked=True)
                .limit(1)
            ).scalar_one_or_none()

            if schedule is None:
                break

            try:
                self._dispatch_and_advance(db, schedule)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "callback_dispatch_error schedule_id=%s: %s",
                    schedule.id,
                    exc,
                    exc_info=True,
                )
                db.rollback()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _dispatch_and_advance(self, db: Session, schedule: CallbackSchedule) -> None:
        """
        1. Verify business hours (reschedule if outside window).
        2. Dispatch the outbound call.
        3. Mark this record executed.
        4. If more attempts remain, insert the next CallbackSchedule.
           Otherwise mark the chain exhausted.
        """
        agent: Optional[Agent] = db.get(Agent, schedule.agent_id)
        if agent is None:
            schedule.status = "cancelled"
            db.commit()
            return

        tz_name = schedule.timezone
        now_tz = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo(tz_name))

        # Re-check business hours at dispatch time
        if not self._is_within_business_hours(db, agent, now_tz):
            # Push to next open window starting from now
            next_window = self._next_valid_window(
                db, agent, datetime.utcnow().replace(tzinfo=ZoneInfo("UTC")), tz_name
            )
            schedule.scheduled_at = next_window
            db.commit()
            logger.info(
                "callback_rescheduled schedule_id=%s next=%s reason=outside_business_hours",
                schedule.id,
                next_window.isoformat(),
            )
            return

        # Dispatch the call
        self._dispatch_call(db, schedule, agent)

        # Mark this attempt executed
        schedule.status = "executed"
        schedule.executed_at = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
        db.add(schedule)

        # Chain next attempt if budget remains
        gap_schedule: List[dict] = agent.callback_gap_schedule or []
        next_attempt_number = schedule.attempt_number + 1

        if next_attempt_number <= agent.max_callback_attempts and gap_schedule:
            # Use the gap at index (next_attempt_number - 1), clamped to last entry
            gap_index = min(next_attempt_number - 1, len(gap_schedule) - 1)
            gap = gap_schedule[gap_index]

            base_time = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
            target_at = base_time + timedelta(
                days=gap.get("days", 0),
                hours=gap.get("hours", 0),
            )
            next_scheduled_at = self._next_valid_window(db, agent, target_at, tz_name)

            next_schedule = CallbackSchedule(
                original_call_id=schedule.original_call_id,
                agent_id=schedule.agent_id,
                phone_number=schedule.phone_number,
                attempt_number=next_attempt_number,
                scheduled_at=next_scheduled_at,
                timezone=tz_name,
                status="pending",
            )
            db.add(next_schedule)
            logger.info(
                "callback_chained original_call_id=%s attempt=%d at=%s",
                schedule.original_call_id,
                next_attempt_number,
                next_scheduled_at.isoformat(),
            )
        else:
            # Budget exhausted — mark this final attempt as exhausted and
            # cancel any stale pending rows that may exist for the same chain.
            schedule.status = "exhausted"
            db.execute(
                CallbackSchedule.__table__.update()
                .where(
                    CallbackSchedule.__table__.c.original_call_id == schedule.original_call_id,
                    CallbackSchedule.__table__.c.status == "pending",
                    CallbackSchedule.__table__.c.id != schedule.id,
                )
                .values(status="cancelled")
            )
            logger.info(
                "callback_exhausted original_call_id=%s max_attempts=%d reached",
                schedule.original_call_id,
                agent.max_callback_attempts,
            )

        db.commit()

    def _dispatch_call(
        self,
        db: Session,
        schedule: CallbackSchedule,
        agent: Agent,
    ) -> None:
        """
        Dispatch the outbound callback call via the existing initiate_call path.

        Session boundaries
        ------------------
        dispatch_db is intentionally separate from the poller session (db).
        initiate_call commits its own work (new CallSession, Twilio request)
        inside dispatch_db. A db.rollback() in the poller NEVER rolls back
        dispatch_db — the call delivery is guaranteed to persist regardless of
        any subsequent bookkeeping failure in the poller session.

        Failure semantics
        -----------------
        - initiate_call failure   → real failure; raise so the poller can
                                    rollback and leave the row in 'pending'.
        - parent_call_id update failure → non-critical bookkeeping; logged
                                          but does NOT fail the dispatch.

        Runs in an APScheduler thread-pool thread (no event loop), so we spin
        up a fresh event loop for the async call.
        """
        import asyncio
        import json as _json

        from fastapi.responses import JSONResponse as _JSONResponse
        from starlette.requests import Request as _Request

        from app.core.config import settings
        from app.db.session import SessionLocal
        from app.schemas.twilio import CallInitiateRequest
        from app.services import voice_call_service as _vcs

        logger.info(
            "callback_dispatch agent_id=%s phone=%s attempt=%d",
            agent.id,
            schedule.phone_number,
            schedule.attempt_number,
        )

        # ── Require webhook secret (same auth path as batch calls) ────────────
        secret = settings.N8N_WEBHOOK_SECRET
        if not secret:
            raise RuntimeError(
                "N8N_WEBHOOK_SECRET must be configured for smart callback dispatch"
            )

        # ── Load the original call to inherit tenant/user context ─────────────
        original = db.get(CallSession, schedule.original_call_id)
        if original is None:
            raise RuntimeError(
                f"Original call {schedule.original_call_id} not found; cannot dispatch callback"
            )

        call_request = CallInitiateRequest(
            agentId=str(agent.id),
            toNumber=schedule.phone_number,
            tenant_id=str(original.tenant_id),
            user_id=str(original.user_id),
        )

        # ── Build a minimal fake Starlette request (same pattern as batch calls) ─
        secret_bytes = secret.encode("latin-1")
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/internal/callback",
            "query_string": b"",
            "headers": [
                (b"x-n8n-webhook-secret", secret_bytes),
                (b"content-type", b"application/json"),
            ],
            "state": {},
        }

        async def _receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        fake_request = _Request(scope, receive=_receive)

        # ── Phase 1: call initiation (dispatch_db, independent of poller db) ────
        # dispatch_db owns its own transaction. Commits inside initiate_call are
        # durable regardless of what happens to the poller session afterwards.
        # APScheduler thread has no running loop, so new_event_loop() is safe.
        dispatch_db = SessionLocal()
        result = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    _vcs.initiate_call(
                        call_request=call_request,
                        http_request=fake_request,
                        user=None,
                        db=dispatch_db,
                    )
                )
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        finally:
            # Always close dispatch_db; its commits are already flushed to PG.
            dispatch_db.close()

        # ── Extract new session id — failure here = real initiation failure ────
        if isinstance(result, _JSONResponse):
            body = _json.loads(result.body)
            new_call_id_str = (body.get("data") or {}).get("callId")
            if not new_call_id_str:
                raise RuntimeError(
                    f"initiate_call failed for callback (attempt {schedule.attempt_number}): "
                    f"{body.get('message', 'unknown error')}"
                )
        else:
            new_call_id_str = getattr(getattr(result, "data", None), "callId", None)

        if not new_call_id_str:
            raise RuntimeError(
                f"initiate_call returned no callId for callback attempt {schedule.attempt_number}"
            )

        logger.info(
            "callback_dispatched new_session=%s parent=%s attempt=%d",
            new_call_id_str,
            schedule.original_call_id,
            schedule.attempt_number,
        )

        # ── Phase 2: bookkeeping — set parent_call_id on the new session ──────
        # Non-critical: the outbound call is already placed. A failure here
        # must NOT propagate as a dispatch failure. Log and continue so the
        # poller can advance the schedule normally.
        new_session_id = uuid.UUID(new_call_id_str)
        try:
            db.execute(
                CallSession.__table__.update()
                .where(CallSession.__table__.c.id == new_session_id)
                .values(parent_call_id=schedule.original_call_id)
            )
            db.commit()
            logger.info(
                "callback_parent_linked new_session=%s parent=%s",
                new_session_id,
                schedule.original_call_id,
            )
        except Exception as link_exc:  # noqa: BLE001
            logger.error(
                "callback_parent_link_failed new_session=%s parent=%s: %s "
                "(call was placed successfully; this is a bookkeeping-only failure)",
                new_session_id,
                schedule.original_call_id,
                link_exc,
                exc_info=True,
            )
            db.rollback()

    # ── Business hours helpers ─────────────────────────────────────────────────

    def _is_within_business_hours(
        self,
        db: Session,
        agent: Agent,
        local_dt: datetime,
    ) -> bool:
        """
        Return True if local_dt falls within the agent's tenant business hours
        for that weekday.  Returns True when no business-hours rows exist
        (open 24/7 by default).
        """
        # BusinessHours.day_of_week: 0 = Monday … 6 = Sunday (matches Python's weekday())
        dow = local_dt.weekday()
        bh: Optional[BusinessHours] = db.execute(
            select(BusinessHours).where(
                BusinessHours.tenant_id == agent.tenant_id,
                BusinessHours.day_of_week == dow,
                BusinessHours.is_deleted == False,  # noqa: E712
            )
        ).scalar_one_or_none()

        if bh is None:
            return True  # No config → treat as always open
        if bh.is_closed:
            return False
        if bh.open_time is None or bh.close_time is None:
            return True

        current_time = local_dt.time()
        return bh.open_time <= current_time <= bh.close_time

    def _next_valid_window(
        self,
        db: Session,
        agent: Agent,
        from_utc: datetime,
        tz_name: str,
    ) -> datetime:
        """
        Starting from from_utc, walk forward in 1-hour increments until a
        slot falls within business hours.  Returns the candidate time in UTC.

        Falls back to from_utc if no open window is found within the lookahead
        period (calendar gap / misconfiguration) to avoid blocking the chain.
        """
        tz = ZoneInfo(tz_name)
        candidate_utc = from_utc

        for _ in range(_MAX_BUSINESS_HOURS_LOOKAHEAD_DAYS * 24):
            local_dt = candidate_utc.astimezone(tz)
            if self._is_within_business_hours(db, agent, local_dt):
                return candidate_utc
            candidate_utc += timedelta(hours=1)

        logger.warning(
            "No open business-hours window found within %d days for agent %s; "
            "using original scheduled time",
            _MAX_BUSINESS_HOURS_LOOKAHEAD_DAYS,
            agent.id,
        )
        return from_utc

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_agent(
        self,
        db: Session,
        agent_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> Agent:
        agent = db.execute(
            select(Agent).where(
                Agent.id == agent_id,
                Agent.tenant_id == tenant_id,
                Agent.is_deleted == False,  # noqa: E712
            )
        ).scalar_one_or_none()

        if agent is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found",
            )
        return agent


callback_scheduler_service = CallbackSchedulerService()
