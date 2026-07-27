from __future__ import annotations

import uuid
from datetime import datetime, timedelta, time
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.core.logger import logger
from app.models.agent import Agent
from app.models.call_flow import CallFlow
from app.models.callback_schedule import CallbackSchedule
from app.models.call_session import CallSession
from app.models.tenant import Tenant
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

# Default calling window when a flow has no business_hours configured
_DEFAULT_OPEN_TIME = time(8, 0)
_DEFAULT_CLOSE_TIME = time(20, 0)

# TCPA hard floor/ceiling for US workspaces — flow-configured hours are
# clamped into this window, never allowed to extend past it.
_TCPA_US_OPEN_TIME = time(8, 0)
_TCPA_US_CLOSE_TIME = time(21, 0)


class InvalidCallbackTimezoneError(Exception):
    """Raised when an agent's callback_timezone is missing or not a valid IANA zone."""

    def __init__(self, agent_id: uuid.UUID):
        self.agent_id = agent_id
        super().__init__(
            f"Agent {agent_id} has an invalid or missing callback_timezone"
        )


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
            gap_schedule=[
                GapInterval(**g) for g in (agent.callback_gap_schedule or [])
            ],
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
            gap_schedule=[
                GapInterval(**g) for g in (agent.callback_gap_schedule or [])
            ],
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

        rows = (
            db.execute(
                select(CallbackSchedule)
                .where(CallbackSchedule.original_call_id == call_id)
                .order_by(CallbackSchedule.attempt_number)
            )
            .scalars()
            .all()
        )

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
        flow, is_us = self._resolve_dispatch_context(db, agent, call_session)
        scheduled_at = self._next_valid_window(
            target_at, tz_name, flow, is_us, agent_id=agent.id
        )

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
        """
        now = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))

        due_rows = (
            db.execute(
                select(CallbackSchedule)
                .where(
                    CallbackSchedule.status == "pending",
                    CallbackSchedule.scheduled_at <= now,
                )
                .with_for_update(skip_locked=True)
            )
            .scalars()
            .all()
        )

        for schedule in due_rows:
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

        try:
            tz_name = self._require_valid_callback_timezone(agent)
        except InvalidCallbackTimezoneError:
            schedule.status = "cancelled"
            db.commit()
            self._alert_invalid_callback_timezone(db, agent)
            logger.error(
                "callback_rejected_invalid_timezone schedule_id=%s agent_id=%s timezone=%r",
                schedule.id,
                agent.id,
                agent.callback_timezone,
            )
            return

        now_utc = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
        now_tz = now_utc.astimezone(ZoneInfo(tz_name))
        flow, is_us = self._resolve_dispatch_context(db, agent, schedule.original_call)

        # Re-check business hours (flow settings + TCPA limits) at dispatch time
        if not self._is_within_business_hours(now_tz, flow, is_us):
            # Push to next open window starting from now
            next_window = self._next_valid_window(
                now_utc, tz_name, flow, is_us, agent_id=agent.id
            )
            original_scheduled_at = schedule.scheduled_at
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
            next_scheduled_at = self._next_valid_window(
                target_at, tz_name, flow, is_us, agent_id=agent.id
            )

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
                    CallbackSchedule.__table__.c.original_call_id
                    == schedule.original_call_id,
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

        Runs in an APScheduler thread-pool thread (no event loop), so we create
        a fresh event loop for the async call and a dedicated DB session so the
        async path can commit without interfering with the caller's session.

        After dispatch the new CallSession.parent_call_id is set to
        schedule.original_call_id for full call-chain traceability.
        """
        import asyncio
        import json as _json

        from fastapi.responses import JSONResponse as _JSONResponse

        from app.db.session import SessionLocal
        from app.schemas.twilio import CallInitiateRequest
        from app.services.voice_call_service import initiate_call

        logger.info(
            "callback_dispatch agent_id=%s phone=%s attempt=%d",
            agent.id,
            schedule.phone_number,
            schedule.attempt_number,
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

        # ── Run async initiate_call in a fresh event loop ─────────────────────
        # APScheduler thread has no running loop, so new_event_loop() is safe.
        dispatch_db = SessionLocal()
        result = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    initiate_call(
                        call_request=call_request,
                        db=dispatch_db,
                        is_system_call=True,
                        tenant_id=original.tenant_id,
                        user_id=original.user_id,
                    )
                )
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        finally:
            dispatch_db.close()

        # ── Extract the new session id and set parent_call_id ─────────────────
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

        if new_call_id_str:
            new_session_id = uuid.UUID(new_call_id_str)
            db.execute(
                CallSession.__table__.update()
                .where(CallSession.__table__.c.id == new_session_id)
                .values(parent_call_id=schedule.original_call_id)
            )
            db.commit()
            logger.info(
                "callback_dispatched new_session=%s parent=%s attempt=%d",
                new_session_id,
                schedule.original_call_id,
                schedule.attempt_number,
            )
        else:
            raise RuntimeError(
                f"initiate_call returned no callId for callback attempt {schedule.attempt_number}"
            )

    # ── ARQ async dispatch (replaces APScheduler polling) ─────────────────────

    async def dispatch_and_advance_async(
        self,
        db: Session,
        schedule: CallbackSchedule,
    ) -> Optional[CallbackSchedule]:
        """
        Async replacement for _dispatch_and_advance, called by the ARQ
        ``execute_callback`` task.

        Returns:
          - The same ``schedule`` (with updated ``scheduled_at``) when it was
            rescheduled due to business-hours constraints — the caller must
            re-enqueue it with the new time.
          - A newly created ``CallbackSchedule`` when the next retry was chained.
          - ``None`` when the retry chain is exhausted or the agent was removed.
        """
        agent: Optional[Agent] = db.get(Agent, schedule.agent_id)
        if agent is None:
            schedule.status = "cancelled"
            db.commit()
            return None

        try:
            tz_name = self._require_valid_callback_timezone(agent)
        except InvalidCallbackTimezoneError:
            schedule.status = "cancelled"
            schedule.arq_job_id = None
            db.commit()
            self._alert_invalid_callback_timezone(db, agent)
            logger.error(
                "callback_rejected_invalid_timezone schedule_id=%s agent_id=%s timezone=%r",
                schedule.id,
                agent.id,
                agent.callback_timezone,
            )
            return None

        now_utc = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
        now_tz = now_utc.astimezone(ZoneInfo(tz_name))
        flow, is_us = self._resolve_dispatch_context(db, agent, schedule.original_call)

        if not self._is_within_business_hours(now_tz, flow, is_us):
            next_window = self._next_valid_window(
                now_utc, tz_name, flow, is_us, agent_id=agent.id
            )
            original_scheduled_at = schedule.scheduled_at
            schedule.scheduled_at = next_window
            # Clear so the recovery cron can re-enqueue with the new time.
            schedule.arq_job_id = None
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
            return schedule

        await self._dispatch_call_async(db, schedule, agent)

        schedule.status = "executed"
        schedule.executed_at = now_utc
        db.add(schedule)

        gap_schedule: List[dict] = agent.callback_gap_schedule or []
        next_attempt_number = schedule.attempt_number + 1

        if next_attempt_number <= agent.max_callback_attempts and gap_schedule:
            gap_index = min(next_attempt_number - 1, len(gap_schedule) - 1)
            gap = gap_schedule[gap_index]
            target_at = now_utc + timedelta(
                days=gap.get("days", 0),
                hours=gap.get("hours", 0),
            )
            next_scheduled_at = self._next_valid_window(
                target_at, tz_name, flow, is_us, agent_id=agent.id
            )

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
            db.commit()
            db.refresh(next_schedule)

            logger.info(
                "callback_chained original_call_id=%s attempt=%d at=%s",
                schedule.original_call_id,
                next_attempt_number,
                next_scheduled_at.isoformat(),
            )
            return next_schedule

        else:
            schedule.status = "exhausted"
            db.execute(
                CallbackSchedule.__table__.update()
                .where(
                    CallbackSchedule.__table__.c.original_call_id
                    == schedule.original_call_id,
                    CallbackSchedule.__table__.c.status == "pending",
                    CallbackSchedule.__table__.c.id != schedule.id,
                )
                .values(status="cancelled")
            )
            db.commit()
            logger.info(
                "callback_exhausted original_call_id=%s max_attempts=%d reached",
                schedule.original_call_id,
                agent.max_callback_attempts,
            )
            return None

    async def _dispatch_call_async(
        self,
        db: Session,
        schedule: CallbackSchedule,
        agent: Agent,
    ) -> None:
        """
        Truly async dispatch — no asyncio.new_event_loop() hack needed.
        Called from the ARQ worker's event loop, so initiate_call is awaited
        directly. A dedicated SessionLocal is used so the async path cannot
        interfere with the caller's sync session.
        """
        import json as _json

        from fastapi.responses import JSONResponse as _JSONResponse

        from app.db.session import SessionLocal
        from app.schemas.twilio import CallInitiateRequest
        from app.services.voice_call_service import initiate_call

        logger.info(
            "callback_dispatch_async agent_id=%s phone=%s attempt=%d",
            agent.id,
            schedule.phone_number,
            schedule.attempt_number,
        )

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

        dispatch_db = SessionLocal()
        try:
            result = await initiate_call(
                call_request=call_request,
                db=dispatch_db,
                is_system_call=True,
                tenant_id=original.tenant_id,
                user_id=original.user_id,
            )
        finally:
            dispatch_db.close()

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

        if new_call_id_str:
            new_session_id = uuid.UUID(new_call_id_str)
            db.execute(
                CallSession.__table__.update()
                .where(CallSession.__table__.c.id == new_session_id)
                .values(parent_call_id=schedule.original_call_id)
            )
            db.commit()
            logger.info(
                "callback_dispatched_async new_session=%s parent=%s attempt=%d",
                new_session_id,
                schedule.original_call_id,
                schedule.attempt_number,
            )
        else:
            raise RuntimeError(
                f"initiate_call returned no callId for callback attempt {schedule.attempt_number}"
            )

    # ── Timezone / TCPA compliance helpers ─────────────────────────────────────

    def _require_valid_callback_timezone(self, agent: Agent) -> str:
        """
        Return agent.callback_timezone if it is a non-empty, valid IANA zone.
        Raises InvalidCallbackTimezoneError otherwise — callers must reject
        the dispatch and alert the workspace admin.
        """
        tz_name = agent.callback_timezone
        if not tz_name:
            raise InvalidCallbackTimezoneError(agent.id)
        try:
            ZoneInfo(tz_name)
        except (ZoneInfoNotFoundError, ValueError):
            raise InvalidCallbackTimezoneError(agent.id)
        return tz_name

    def _alert_invalid_callback_timezone(self, db: Session, agent: Agent) -> None:
        """
        Notify the workspace admin that this agent's callback_timezone is
        missing/invalid, blocking compliant callback dispatch.
        """
        from app.services.data_export_service import _get_workspace_admin_email
        from app.services.email_service import email_service

        admin_email = _get_workspace_admin_email(db, agent.tenant_id)
        if not admin_email:
            tenant = db.get(Tenant, agent.tenant_id)
            admin_email = getattr(tenant, "contact_email", None) if tenant else None

        if not admin_email:
            logger.warning(
                "No admin or contact email found for tenant %s; cannot send "
                "invalid-timezone alert for agent %s",
                agent.tenant_id,
                agent.id,
            )
            return

        email_service.send_generic_email(
            to_email=admin_email,
            subject="Action required: invalid callback timezone is blocking compliant callbacks",
            html_body=(
                f"<p>Agent <strong>{agent.id}</strong> has an invalid or missing "
                f"<code>callback_timezone</code> (current value: {agent.callback_timezone!r}). "
                "Scheduled callbacks for this agent cannot be dispatched until a valid "
                "IANA timezone is configured, in order to remain TCPA-compliant.</p>"
            ),
        )

    def _is_us_workspace(self, tenant: Optional[Tenant]) -> bool:
        if tenant is None or not tenant.workspace_settings:
            return False
        ws = tenant.workspace_settings
        return ws.get("country") == "US" or ws.get("workspace_country") == "US"

    def _get_flow_for_dispatch(
        self,
        db: Session,
        agent: Agent,
        original_call,
    ) -> Optional[CallFlow]:
        """
        Resolve the CallFlow whose settings.business_hours governs this
        callback: prefer the flow the original call ran on, falling back to
        any active flow belonging to the agent.
        """
        flow: Optional[CallFlow] = None
        call_flow_id = (
            getattr(original_call, "call_flow_id", None)
            if original_call is not None
            else None
        )
        if call_flow_id:
            flow = db.get(CallFlow, call_flow_id)

        if flow is None:
            flow = (
                db.execute(
                    select(CallFlow)
                    .where(
                        CallFlow.agent_id == agent.id,
                        CallFlow.is_deleted == False,  # noqa: E712
                    )
                    .order_by(CallFlow.created_at.asc())
                )
                .scalars()
                .first()
            )

        return flow

    def _resolve_dispatch_context(
        self,
        db: Session,
        agent: Agent,
        original_call,
    ) -> Tuple[Optional[CallFlow], bool]:
        """
        Resolve the (flow, is_us_workspace) pair once per dispatch so the
        business-hours check and the reschedule computation agree on the
        exact same flow/workspace state instead of re-querying separately.
        """
        tenant = db.get(Tenant, agent.tenant_id)
        is_us = self._is_us_workspace(tenant)
        flow = self._get_flow_for_dispatch(db, agent, original_call)
        return flow, is_us

    def _lookup_day_config(self, business_hours, dow: int) -> Optional[dict]:
        if isinstance(business_hours, dict):
            cfg = business_hours.get(str(dow))
            if cfg is None:
                cfg = business_hours.get(dow)
            return cfg
        if isinstance(business_hours, list):
            for entry in business_hours:
                if isinstance(entry, dict) and entry.get("day") == dow:
                    return entry
        return None

    def _parse_time(self, value) -> Optional[time]:
        if value is None:
            return None
        if isinstance(value, time):
            return value
        try:
            parts = str(value).split(":")
            return time(int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
        except (ValueError, IndexError):
            return None

    def _resolve_day_window(
        self,
        flow: Optional[CallFlow],
        dow: int,
        is_us: bool,
    ) -> Optional[Tuple[time, time]]:
        """
        Return the (open, close) local-time window for the given weekday
        (0=Monday…6=Sunday), or None if the day is closed.

        Falls back to the 8am-8pm default when the flow has no
        business_hours configured. US workspaces are additionally clamped
        to the 8am-9pm TCPA hard limit.
        """
        business_hours = None
        if flow is not None and flow.settings:
            business_hours = flow.settings.get("business_hours")

        if not business_hours:
            open_t, close_t = _DEFAULT_OPEN_TIME, _DEFAULT_CLOSE_TIME
        else:
            day_cfg = self._lookup_day_config(business_hours, dow)
            if day_cfg is None:
                open_t, close_t = _DEFAULT_OPEN_TIME, _DEFAULT_CLOSE_TIME
            elif day_cfg.get("closed"):
                return None
            else:
                open_t = self._parse_time(day_cfg.get("open")) or _DEFAULT_OPEN_TIME
                close_t = self._parse_time(day_cfg.get("close")) or _DEFAULT_CLOSE_TIME

        if is_us:
            if open_t < _TCPA_US_OPEN_TIME:
                open_t = _TCPA_US_OPEN_TIME
            if close_t > _TCPA_US_CLOSE_TIME:
                close_t = _TCPA_US_CLOSE_TIME
            if open_t >= close_t:
                return None

        return (open_t, close_t)

    # ── Business hours helpers ─────────────────────────────────────────────────

    def _is_within_business_hours(
        self,
        local_dt: datetime,
        flow: Optional[CallFlow],
        is_us: bool,
    ) -> bool:
        """
        Return True if local_dt falls within the flow-configured business
        hours (TCPA-clamped for US workspaces) for that weekday.
        """
        window = self._resolve_day_window(flow, local_dt.weekday(), is_us)
        if window is None:
            return False

        open_t, close_t = window
        return open_t <= local_dt.time() <= close_t

    def _next_valid_window(
        self,
        from_utc: datetime,
        tz_name: str,
        flow: Optional[CallFlow],
        is_us: bool,
        agent_id: Optional[uuid.UUID] = None,
    ) -> datetime:
        """
        Resolve the next compliant dispatch time:
        - if before today's open time, use today's open time
        - if after today's close time (or today is closed), use the next
          available open day's open time
        Returns the candidate time in UTC.

        If no open window exists within the lookahead period (e.g. every
        day is configured closed), pushes the candidate forward by the
        full lookahead window rather than returning from_utc unchanged —
        returning it as-is would leave the schedule permanently outside
        business hours, causing the poller to reschedule it to the same
        non-compliant instant on every tick forever.
        """
        tz = ZoneInfo(tz_name)
        local_dt = from_utc.astimezone(tz)
        window = self._resolve_day_window(flow, local_dt.weekday(), is_us)

        if window is not None:
            open_t, close_t = window
            if local_dt.time() < open_t:
                target_local = local_dt.replace(
                    hour=open_t.hour, minute=open_t.minute, second=0, microsecond=0
                )
                return target_local.astimezone(ZoneInfo("UTC"))
            if local_dt.time() <= close_t:
                return from_utc  # already within window

        for day_offset in range(1, _MAX_BUSINESS_HOURS_LOOKAHEAD_DAYS + 1):
            candidate_local = local_dt + timedelta(days=day_offset)
            candidate_window = self._resolve_day_window(
                flow, candidate_local.weekday(), is_us
            )
            if candidate_window is not None:
                open_t, _close_t = candidate_window
                target_local = candidate_local.replace(
                    hour=open_t.hour, minute=open_t.minute, second=0, microsecond=0
                )
                return target_local.astimezone(ZoneInfo("UTC"))

        logger.warning(
            "No compliant business-hours window found within %d days for agent %s "
            "(flow misconfigured with no open days?); pushing candidate forward by "
            "the lookahead window instead of leaving it non-compliant",
            _MAX_BUSINESS_HOURS_LOOKAHEAD_DAYS,
            agent_id,
        )
        return from_utc + timedelta(days=_MAX_BUSINESS_HOURS_LOOKAHEAD_DAYS)

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
