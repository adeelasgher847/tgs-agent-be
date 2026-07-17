"""
Tests for the Smart Callback Scheduler feature.

Coverage:
  1. no_answer call correctly triggers callback creation
  2. busy call correctly triggers callback creation
  3. Disabled agent does NOT create callback
  4. Invalid timezone returns 422 on PUT /agents/{id}/callback-config
  5. Valid timezone is accepted
  6. GET /agents/{id}/callback-status returns correct counters
  7. GET /calls/{call_id}/callback-history returns ordered history
  8. Business hours (flow settings): 3am caller-local reschedules to next open window
  9. Business hours: open window is used as-is (valid timezone allows on-hours callback)
 10. Exhaustion: no further schedule is created at max_attempts
 11. Exhausted schedule is logged with status='exhausted'
 12. Gap schedule index is clamped at last entry when attempts exceed schedule length
 13. get_callback_history returns 404 for unknown call
 14. get_callback_history returns 404 for cross-tenant call
 15. PUT callback-config returns 404 for unknown agent
 16. Missing/invalid callback_timezone rejects dispatch and sends an admin alert email
 17. US workspace clamps flow hours to the 8am-9pm TCPA hard limit
"""

from __future__ import annotations

import uuid
from datetime import datetime, time, timedelta
from typing import Optional
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from app.models.agent import Agent
from app.models.call_flow import CallFlow
from app.models.callback_schedule import CallbackSchedule
from app.models.call_session import CallSession
from app.models.tenant import Tenant
from app.schemas.callback_scheduler import (
    CallbackConfigUpdate,
    GapInterval,
)
from app.services.callback_scheduler_service import (
    CallbackSchedulerService,
    CALLBACK_TRIGGER_STATUSES,
    InvalidCallbackTimezoneError,
    _MAX_BUSINESS_HOURS_LOOKAHEAD_DAYS,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_agent(
    *,
    smart_callback_enabled: bool = True,
    max_callback_attempts: int = 3,
    gap_schedule: Optional[list] = None,
    callback_timezone: str = "UTC",
    tenant_id: Optional[uuid.UUID] = None,
) -> Agent:
    agent = MagicMock(spec=Agent)
    agent.id = uuid.uuid4()
    agent.tenant_id = tenant_id or uuid.uuid4()
    agent.smart_callback_enabled = smart_callback_enabled
    agent.max_callback_attempts = max_callback_attempts
    agent.callback_gap_schedule = gap_schedule or [{"days": 0, "hours": 1}]
    agent.callback_timezone = callback_timezone
    agent.is_deleted = False
    return agent


def _make_call_session(
    *,
    status: str = "no_answer",
    agent_id: Optional[uuid.UUID] = None,
    tenant_id: Optional[uuid.UUID] = None,
    to_number: str = "+15550001234",
    customer_phone_number: Optional[str] = None,
    call_flow_id: Optional[uuid.UUID] = None,
) -> CallSession:
    cs = MagicMock(spec=CallSession)
    cs.id = uuid.uuid4()
    cs.agent_id = agent_id or uuid.uuid4()
    cs.tenant_id = tenant_id or uuid.uuid4()
    cs.status = status
    cs.to_number = to_number
    cs.customer_phone_number = customer_phone_number or to_number
    cs.user_id = uuid.uuid4()
    cs.call_flow_id = call_flow_id
    return cs


def _make_flow(
    *,
    agent_id: Optional[uuid.UUID] = None,
    business_hours: Optional[dict] = None,
) -> CallFlow:
    flow = MagicMock(spec=CallFlow)
    flow.id = uuid.uuid4()
    flow.agent_id = agent_id or uuid.uuid4()
    flow.is_deleted = False
    flow.settings = (
        {"business_hours": business_hours} if business_hours is not None else None
    )
    return flow


def _make_tenant(*, is_us: bool = False, contact_email: Optional[str] = None) -> Tenant:
    tenant = MagicMock(spec=Tenant)
    tenant.id = uuid.uuid4()
    tenant.workspace_settings = {"country": "US"} if is_us else {}
    tenant.contact_email = contact_email
    return tenant


def _make_dispatch_db(
    *,
    agent: Agent,
    original_call: CallSession,
    tenant: Optional[Tenant] = None,
    flow: Optional[CallFlow] = None,
) -> MagicMock:
    """
    Mock Session for exercising _dispatch_and_advance / dispatch_and_advance_async:
    resolves Agent/CallSession/Tenant via db.get, and CallFlow lookups
    (both direct id and the agent-scoped fallback query) via db.execute.
    """
    db = MagicMock()

    def _get(model_cls, pk):
        if model_cls is Agent:
            return agent
        if model_cls is CallSession:
            return original_call
        if model_cls is Tenant:
            return tenant
        if model_cls is CallFlow:
            return flow if original_call.call_flow_id else None
        return None

    db.get.side_effect = _get

    execute_result = MagicMock()
    execute_result.scalars.return_value.first.return_value = flow
    execute_result.scalars.return_value.all.return_value = []
    db.execute.return_value = execute_result

    return db


def _make_db(
    *,
    agent: Optional[Agent] = None,
    call_session: Optional[CallSession] = None,
    existing_schedules: Optional[list] = None,
) -> MagicMock:
    """Return a mock Session with sensible defaults."""
    db = MagicMock()

    def _get(model_cls, pk):
        if model_cls is Agent and agent and agent.id == pk:
            return agent
        if model_cls is CallSession and call_session and call_session.id == pk:
            return call_session
        return None

    db.get.side_effect = _get

    # Mock execute().scalar_one_or_none() chain
    execute_result = MagicMock()
    scalar_result = MagicMock()

    if agent:
        scalar_result.scalar_one_or_none.return_value = agent
    else:
        scalar_result.scalar_one_or_none.return_value = None

    scalar_result.scalar_one.return_value = 0
    scalar_result.scalars.return_value.all.return_value = existing_schedules or []
    execute_result.scalar_one_or_none.return_value = agent
    execute_result.scalar_one.return_value = 0
    execute_result.scalars.return_value.all.return_value = existing_schedules or []

    db.execute.return_value = execute_result
    return db


# ── 1. no_answer triggers callback ────────────────────────────────────────────


def test_no_answer_triggers_callback_creation():
    """A call ending as 'no_answer' should insert a CallbackSchedule."""
    agent = _make_agent()
    call = _make_call_session(status="no_answer", agent_id=agent.id)

    db = MagicMock()
    db.get.side_effect = lambda cls, pk: agent if cls is Agent else None

    svc = CallbackSchedulerService()
    with patch("app.services.callback_scheduler_service.datetime") as mock_dt:
        mock_dt.utcnow.return_value = datetime(2026, 6, 15, 10, 0, 0)
        mock_dt.utcnow.return_value = mock_dt.utcnow.return_value.replace(
            tzinfo=ZoneInfo("UTC")
        )
        # Make the _next_valid_window return immediately (no BH rows)
        db.execute.return_value.scalar_one_or_none.return_value = None

        result = svc.maybe_schedule_callback(db, call)

    db.add.assert_called_once()
    db.commit.assert_called_once()
    added = db.add.call_args[0][0]
    assert isinstance(added, CallbackSchedule)
    assert added.original_call_id == call.id
    assert added.agent_id == agent.id
    assert added.attempt_number == 1
    assert added.status == "pending"
    assert added.phone_number == call.customer_phone_number


# ── 2. busy triggers callback ─────────────────────────────────────────────────


def test_busy_triggers_callback_creation():
    """A call ending as 'busy' should also insert a CallbackSchedule."""
    agent = _make_agent()
    call = _make_call_session(status="busy", agent_id=agent.id)

    db = MagicMock()
    db.get.side_effect = lambda cls, pk: agent if cls is Agent else None
    db.execute.return_value.scalar_one_or_none.return_value = None

    svc = CallbackSchedulerService()
    result = svc.maybe_schedule_callback(db, call)

    db.add.assert_called_once()
    added = db.add.call_args[0][0]
    assert added.attempt_number == 1
    assert added.phone_number == call.customer_phone_number


# ── 3. Disabled agent does not create callback ────────────────────────────────


def test_disabled_agent_does_not_schedule():
    """smart_callback_enabled=False must short-circuit without any DB write."""
    agent = _make_agent(smart_callback_enabled=False)
    call = _make_call_session(status="no_answer", agent_id=agent.id)

    db = MagicMock()
    db.get.side_effect = lambda cls, pk: agent if cls is Agent else None

    svc = CallbackSchedulerService()
    result = svc.maybe_schedule_callback(db, call)

    assert result is None
    db.add.assert_not_called()


# ── 4. Non-triggering status does not create callback ─────────────────────────


def test_completed_call_does_not_trigger_callback():
    """Calls that completed normally must not trigger the callback chain."""
    agent = _make_agent()
    call = _make_call_session(status="completed", agent_id=agent.id)

    db = MagicMock()
    svc = CallbackSchedulerService()
    result = svc.maybe_schedule_callback(db, call)

    assert result is None
    db.add.assert_not_called()
    db.get.assert_not_called()


# ── 5. Invalid timezone returns 422 ───────────────────────────────────────────


def test_invalid_timezone_raises_validation_error():
    """CallbackConfigUpdate must raise ValidationError for unknown timezone."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc_info:
        CallbackConfigUpdate(
            smart_callback_enabled=True,
            max_attempts=3,
            gap_schedule=[{"days": 0, "hours": 1}],
            timezone="Narnia/Castle",
        )

    errors = exc_info.value.errors()
    assert any(
        "timezone" in str(e.get("loc", "")) or "timezone" in str(e) for e in errors
    )


# ── 6. Valid timezone is accepted ─────────────────────────────────────────────


def test_valid_timezone_is_accepted():
    """Well-known IANA timezones must pass the validator."""
    cfg = CallbackConfigUpdate(
        smart_callback_enabled=True,
        max_attempts=5,
        gap_schedule=[{"days": 0, "hours": 2}],
        timezone="America/New_York",
    )
    assert cfg.timezone == "America/New_York"


# ── 7. gap_schedule must be non-empty when enabled ───────────────────────────


def test_empty_gap_schedule_raises_when_enabled():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CallbackConfigUpdate(
            smart_callback_enabled=True,
            max_attempts=3,
            gap_schedule=[],
            timezone="UTC",
        )


# ── 8. gap_schedule length cannot exceed max_attempts ────────────────────────


def test_gap_schedule_exceeds_max_attempts_raises():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CallbackConfigUpdate(
            smart_callback_enabled=True,
            max_attempts=2,
            gap_schedule=[
                {"days": 0, "hours": 1},
                {"days": 0, "hours": 2},
                {"days": 0, "hours": 3},
            ],
            timezone="UTC",
        )


# ── 9. GET callback-status returns correct counters ───────────────────────────


def test_get_callback_status_returns_counters():
    """Pending count and next_scheduled_at are fetched from DB."""
    agent = _make_agent()
    future_dt = datetime(2026, 6, 16, 9, 0, 0, tzinfo=ZoneInfo("UTC"))

    db = MagicMock()

    # First execute call: agent lookup
    # Second execute call: count query  → 3
    # Third execute call: min(scheduled_at) → future_dt
    call_count = [0]

    def _execute(stmt, *args, **kwargs):
        call_count[0] += 1
        m = MagicMock()
        if call_count[0] == 1:
            m.scalar_one_or_none.return_value = agent
        elif call_count[0] == 2:
            m.scalar_one.return_value = 3
        else:
            m.scalar_one.return_value = future_dt
        return m

    db.execute.side_effect = _execute

    svc = CallbackSchedulerService()
    result = svc.get_callback_status(db, agent.id, agent.tenant_id)

    assert result.enabled == agent.smart_callback_enabled
    assert result.pending_retries == 3
    assert result.next_scheduled_at == future_dt


# ── 10. GET callback-history returns ordered list ────────────────────────────


def test_get_callback_history_returns_ordered_rows():
    """History endpoint returns all schedule rows for the original call."""
    tenant_id = uuid.uuid4()
    original_call_id = uuid.uuid4()

    call = _make_call_session(tenant_id=tenant_id)
    call.id = original_call_id

    sched1 = MagicMock(spec=CallbackSchedule)
    sched1.attempt_number = 1
    sched1.scheduled_at = datetime(2026, 6, 15, 10, 0, tzinfo=ZoneInfo("UTC"))
    sched1.executed_at = datetime(2026, 6, 15, 10, 30, tzinfo=ZoneInfo("UTC"))
    sched1.status = "executed"

    sched2 = MagicMock(spec=CallbackSchedule)
    sched2.attempt_number = 2
    sched2.scheduled_at = datetime(2026, 6, 16, 10, 0, tzinfo=ZoneInfo("UTC"))
    sched2.executed_at = None
    sched2.status = "pending"

    call_count = [0]

    def _execute(stmt, *args, **kwargs):
        call_count[0] += 1
        m = MagicMock()
        if call_count[0] == 1:
            m.scalar_one_or_none.return_value = call
        else:
            m.scalars.return_value.all.return_value = [sched1, sched2]
        return m

    db = MagicMock()
    db.execute.side_effect = _execute

    svc = CallbackSchedulerService()
    history = svc.get_callback_history(db, original_call_id, tenant_id)

    assert len(history) == 2
    assert history[0].attempt_number == 1
    assert history[0].status == "executed"
    assert history[1].attempt_number == 2
    assert history[1].status == "pending"


# ── 11. Business hours check: outside window reschedules ─────────────────────


def test_outside_business_hours_reschedules():
    """
    A callback scheduled at 3am in the caller's (agent's callback_timezone)
    local time must be rescheduled to the flow's next business-hours open
    time, and must NOT be dispatched.
    """
    tz_name = "America/New_York"
    agent = _make_agent(callback_timezone=tz_name)
    # 07:00 UTC == 03:00 Eastern (outside the flow's configured 09:00-17:00)
    closed_dt = datetime(2026, 6, 15, 7, 0, 0, tzinfo=ZoneInfo("UTC"))
    dow = closed_dt.astimezone(ZoneInfo(tz_name)).weekday()

    flow = _make_flow(
        agent_id=agent.id,
        business_hours={str(dow): {"open": "09:00", "close": "17:00"}},
    )

    original_call = _make_call_session(call_flow_id=flow.id)
    schedule = MagicMock(spec=CallbackSchedule)
    schedule.id = uuid.uuid4()
    schedule.original_call_id = original_call.id
    schedule.original_call = original_call
    schedule.agent_id = agent.id
    schedule.phone_number = "+15550001234"
    schedule.attempt_number = 1
    schedule.scheduled_at = closed_dt
    schedule.timezone = tz_name

    db = _make_dispatch_db(agent=agent, original_call=original_call, flow=flow)

    svc = CallbackSchedulerService()

    with patch("app.services.callback_scheduler_service.datetime") as mock_dt:
        mock_dt.utcnow.return_value = closed_dt.replace(tzinfo=None)
        svc._dispatch_and_advance(db, schedule)

    # The schedule should have been rescheduled to today's 09:00 Eastern open time
    assert schedule.status != "executed"
    new_local = schedule.scheduled_at.astimezone(ZoneInfo(tz_name))
    assert new_local.time() == time(9, 0)
    assert new_local.date() == closed_dt.astimezone(ZoneInfo(tz_name)).date()
    db.commit.assert_called()


# ── 12. Within business hours: call is dispatched ────────────────────────────


def test_within_business_hours_dispatches_call():
    """
    A valid timezone with the current local time inside the flow's
    business hours must dispatch the call and mark the schedule executed.
    """
    agent = _make_agent(
        max_callback_attempts=3,
        gap_schedule=[
            {"days": 0, "hours": 1},
            {"days": 0, "hours": 2},
            {"days": 1, "hours": 0},
        ],
        callback_timezone="UTC",
    )
    open_dt = datetime(2026, 6, 15, 14, 0, 0, tzinfo=ZoneInfo("UTC"))
    dow = open_dt.weekday()

    flow = _make_flow(
        agent_id=agent.id,
        business_hours={str(dow): {"open": "09:00", "close": "17:00"}},
    )

    original_call = _make_call_session(agent_id=agent.id, call_flow_id=flow.id)
    original_call.id = uuid.uuid4()
    original_call.user_id = uuid.uuid4()
    original_call.tenant_id = uuid.uuid4()

    schedule = MagicMock(spec=CallbackSchedule)
    schedule.id = uuid.uuid4()
    schedule.original_call_id = original_call.id
    schedule.original_call = original_call
    schedule.agent_id = agent.id
    schedule.phone_number = "+15550001234"
    schedule.attempt_number = 1
    schedule.scheduled_at = open_dt
    schedule.timezone = "UTC"
    schedule.status = "pending"

    db = _make_dispatch_db(agent=agent, original_call=original_call, flow=flow)

    svc = CallbackSchedulerService()

    with (
        patch("app.services.callback_scheduler_service.datetime") as mock_dt,
        patch.object(svc, "_dispatch_call") as mock_dispatch,
    ):
        mock_dt.utcnow.return_value = open_dt
        svc._dispatch_and_advance(db, schedule)

    mock_dispatch.assert_called_once_with(db, schedule, agent)

    # Schedule must be marked executed
    assert schedule.status == "executed"
    assert schedule.executed_at is not None
    # A new (chained) CallbackSchedule should have been added
    add_calls = [
        call_args[0][0]
        for call_args in db.add.call_args_list
        if isinstance(call_args[0][0], CallbackSchedule)
        and call_args[0][0] is not schedule
    ]
    assert len(add_calls) == 1, "Expected one chained CallbackSchedule to be added"
    assert add_calls[0].attempt_number == 2


# ── 13. Exhaustion: no further schedule at max_attempts ───────────────────────


def test_exhaustion_at_max_attempts():
    """
    When attempt_number == max_callback_attempts, status becomes 'exhausted'
    and no further CallbackSchedule is inserted.
    """
    agent = _make_agent(
        max_callback_attempts=2,
        gap_schedule=[{"days": 0, "hours": 1}],
        callback_timezone="UTC",
    )
    # 10:00 UTC falls within the default 8am-8pm window (no flow configured)
    open_dt = datetime(2026, 6, 15, 10, 0, 0, tzinfo=ZoneInfo("UTC"))

    original_call = _make_call_session(agent_id=agent.id, call_flow_id=None)
    original_call.id = uuid.uuid4()
    original_call.user_id = uuid.uuid4()
    original_call.tenant_id = uuid.uuid4()

    schedule = MagicMock(spec=CallbackSchedule)
    schedule.id = uuid.uuid4()
    schedule.original_call_id = original_call.id
    schedule.original_call = original_call
    schedule.agent_id = agent.id
    schedule.phone_number = "+15550001234"
    # This is the LAST allowed attempt
    schedule.attempt_number = 2
    schedule.scheduled_at = open_dt
    schedule.timezone = "UTC"
    schedule.status = "pending"

    db = _make_dispatch_db(agent=agent, original_call=original_call, flow=None)

    svc = CallbackSchedulerService()

    with (
        patch("app.services.callback_scheduler_service.datetime") as mock_dt,
        patch.object(svc, "_dispatch_call") as mock_dispatch,
    ):
        mock_dt.utcnow.return_value = open_dt
        svc._dispatch_and_advance(db, schedule)

    mock_dispatch.assert_called_once_with(db, schedule, agent)

    # Final schedule must be exhausted, no new record added
    assert schedule.status == "exhausted"
    chained = [
        call_args[0][0]
        for call_args in db.add.call_args_list
        if isinstance(call_args[0][0], CallbackSchedule)
        and call_args[0][0] is not schedule
    ]
    assert len(chained) == 0, "No chained schedule should be created after exhaustion"


# ── 14. get_callback_history: 404 for unknown call ───────────────────────────


def test_callback_history_404_for_unknown_call():
    from fastapi import HTTPException

    db = MagicMock()
    db.execute.return_value.scalar_one_or_none.return_value = None

    svc = CallbackSchedulerService()
    with pytest.raises(HTTPException) as exc_info:
        svc.get_callback_history(db, uuid.uuid4(), uuid.uuid4())

    assert exc_info.value.status_code == 404


# ── 15. update_callback_config: 404 for unknown agent ────────────────────────


def test_update_config_404_for_unknown_agent():
    from fastapi import HTTPException

    db = MagicMock()
    db.execute.return_value.scalar_one_or_none.return_value = None

    svc = CallbackSchedulerService()
    cfg = CallbackConfigUpdate(
        smart_callback_enabled=True,
        max_attempts=3,
        gap_schedule=[{"days": 0, "hours": 1}],
        timezone="UTC",
    )
    with pytest.raises(HTTPException) as exc_info:
        svc.update_callback_config(db, uuid.uuid4(), uuid.uuid4(), cfg)

    assert exc_info.value.status_code == 404


# ── 16. GapInterval zero-interval validation ──────────────────────────────────


def test_gap_interval_zero_raises():
    """Both days=0 and hours=0 is meaningless; must raise ValidationError."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GapInterval(days=0, hours=0)


# ── 17. maybe_schedule_callback: no phone number skips ───────────────────────


def test_no_phone_number_skips_schedule():
    """If the call has no destination number, no schedule should be created."""
    agent = _make_agent()
    call = _make_call_session(
        status="no_answer",
        agent_id=agent.id,
        to_number=None,
        customer_phone_number=None,
    )
    call.to_number = None
    call.customer_phone_number = None

    db = MagicMock()
    db.get.side_effect = lambda cls, pk: agent if cls is Agent else None
    db.execute.return_value.scalar_one_or_none.return_value = None

    svc = CallbackSchedulerService()
    result = svc.maybe_schedule_callback(db, call)

    assert result is None
    db.add.assert_not_called()


# ── 18. CALLBACK_TRIGGER_STATUSES constant coverage ──────────────────────────


def test_trigger_status_set_contents():
    """Verify exactly the statuses that trigger a callback."""
    assert "no_answer" in CALLBACK_TRIGGER_STATUSES
    assert "busy" in CALLBACK_TRIGGER_STATUSES
    assert "completed" not in CALLBACK_TRIGGER_STATUSES
    assert "failed" not in CALLBACK_TRIGGER_STATUSES


# ── 19. Missing/invalid timezone rejects dispatch and alerts admin ────────────


@pytest.mark.parametrize("bad_timezone", [None, "", "Not/AZone"])
def test_require_valid_callback_timezone_raises(bad_timezone):
    """_require_valid_callback_timezone must reject missing/invalid IANA zones."""
    agent = _make_agent(callback_timezone=bad_timezone)
    svc = CallbackSchedulerService()
    with pytest.raises(InvalidCallbackTimezoneError):
        svc._require_valid_callback_timezone(agent)


def test_require_valid_callback_timezone_accepts_valid_zone():
    agent = _make_agent(callback_timezone="America/Los_Angeles")
    svc = CallbackSchedulerService()
    assert svc._require_valid_callback_timezone(agent) == "America/Los_Angeles"


@pytest.mark.parametrize("bad_timezone", [None, "", "Not/AZone"])
def test_invalid_timezone_rejects_dispatch_and_sends_alert(bad_timezone):
    """
    An agent with a missing or invalid callback_timezone must never dispatch
    a callback; the schedule is cancelled and the workspace admin is emailed.
    """
    agent = _make_agent(callback_timezone=bad_timezone)
    original_call = _make_call_session(agent_id=agent.id, call_flow_id=None)

    schedule = MagicMock(spec=CallbackSchedule)
    schedule.id = uuid.uuid4()
    schedule.original_call_id = original_call.id
    schedule.original_call = original_call
    schedule.agent_id = agent.id
    schedule.scheduled_at = datetime(2026, 6, 15, 10, 0, 0, tzinfo=ZoneInfo("UTC"))
    schedule.status = "pending"

    db = _make_dispatch_db(agent=agent, original_call=original_call, flow=None)

    svc = CallbackSchedulerService()

    with (
        patch(
            "app.services.data_export_service._get_workspace_admin_email",
            return_value="admin@example.com",
        ),
        patch(
            "app.services.email_service.email_service.send_generic_email"
        ) as mock_send,
        patch.object(svc, "_dispatch_call") as mock_dispatch,
    ):
        svc._dispatch_and_advance(db, schedule)

    mock_dispatch.assert_not_called()
    assert schedule.status == "cancelled"
    mock_send.assert_called_once()
    assert mock_send.call_args.kwargs["to_email"] == "admin@example.com"


def test_invalid_timezone_falls_back_to_tenant_contact_email():
    """When no admin user exists, the alert falls back to tenant.contact_email."""
    agent = _make_agent(callback_timezone=None)
    original_call = _make_call_session(agent_id=agent.id, call_flow_id=None)
    tenant = _make_tenant(contact_email="owner@example.com")

    schedule = MagicMock(spec=CallbackSchedule)
    schedule.id = uuid.uuid4()
    schedule.original_call_id = original_call.id
    schedule.original_call = original_call
    schedule.agent_id = agent.id
    schedule.scheduled_at = datetime(2026, 6, 15, 10, 0, 0, tzinfo=ZoneInfo("UTC"))
    schedule.status = "pending"

    db = _make_dispatch_db(
        agent=agent, original_call=original_call, tenant=tenant, flow=None
    )

    svc = CallbackSchedulerService()

    with (
        patch(
            "app.services.data_export_service._get_workspace_admin_email",
            return_value=None,
        ),
        patch(
            "app.services.email_service.email_service.send_generic_email"
        ) as mock_send,
    ):
        svc._dispatch_and_advance(db, schedule)

    assert schedule.status == "cancelled"
    mock_send.assert_called_once()
    assert mock_send.call_args.kwargs["to_email"] == "owner@example.com"


# ── 20. US workspace clamps flow hours to the TCPA 8am-9pm hard limit ─────────


def test_us_workspace_clamps_to_tcpa_hard_limit():
    """
    A flow configured for 06:00-22:00 in a US workspace must be clamped to
    08:00-21:00; a callback at 21:30 local (inside the flow window but
    outside the TCPA ceiling) must be rescheduled, not dispatched.
    """
    tz_name = "UTC"
    agent = _make_agent(callback_timezone=tz_name)
    late_dt = datetime(2026, 6, 15, 21, 30, 0, tzinfo=ZoneInfo("UTC"))
    dow = late_dt.weekday()

    flow = _make_flow(
        agent_id=agent.id,
        business_hours={str(dow): {"open": "06:00", "close": "22:00"}},
    )
    tenant = _make_tenant(is_us=True)

    original_call = _make_call_session(agent_id=agent.id, call_flow_id=flow.id)
    schedule = MagicMock(spec=CallbackSchedule)
    schedule.id = uuid.uuid4()
    schedule.original_call_id = original_call.id
    schedule.original_call = original_call
    schedule.agent_id = agent.id
    schedule.phone_number = "+15550001234"
    schedule.attempt_number = 1
    schedule.scheduled_at = late_dt
    schedule.timezone = tz_name

    db = _make_dispatch_db(
        agent=agent, original_call=original_call, tenant=tenant, flow=flow
    )

    svc = CallbackSchedulerService()

    with (
        patch("app.services.callback_scheduler_service.datetime") as mock_dt,
        patch.object(svc, "_dispatch_call") as mock_dispatch,
    ):
        mock_dt.utcnow.return_value = late_dt.replace(tzinfo=None)
        svc._dispatch_and_advance(db, schedule)

    mock_dispatch.assert_not_called()
    assert schedule.status != "executed"
    # Rescheduled to the next compliant open time (08:00, next day since today
    # is already past the 21:00 TCPA ceiling)
    new_local = schedule.scheduled_at.astimezone(ZoneInfo(tz_name))
    assert new_local.time() == time(8, 0)
    assert new_local.date() > late_dt.date()


# ── 21. All-days-closed misconfiguration never yields a non-advancing reschedule ──


def test_all_days_closed_pushes_forward_instead_of_hot_looping():
    """
    A flow misconfigured with every day closed must never leave the
    schedule pinned at the original (non-compliant) instant — that would
    make the poller re-select and re-reschedule the same row forever.
    _next_valid_window must push the candidate strictly into the future.
    """
    tz_name = "UTC"
    agent = _make_agent(callback_timezone=tz_name)
    now_dt = datetime(2026, 6, 15, 10, 0, 0, tzinfo=ZoneInfo("UTC"))

    flow = _make_flow(
        agent_id=agent.id,
        business_hours={str(d): {"closed": True} for d in range(7)},
    )

    original_call = _make_call_session(agent_id=agent.id, call_flow_id=flow.id)
    schedule = MagicMock(spec=CallbackSchedule)
    schedule.id = uuid.uuid4()
    schedule.original_call_id = original_call.id
    schedule.original_call = original_call
    schedule.agent_id = agent.id
    schedule.phone_number = "+15550001234"
    schedule.attempt_number = 1
    schedule.scheduled_at = now_dt
    schedule.timezone = tz_name

    db = _make_dispatch_db(agent=agent, original_call=original_call, flow=flow)

    svc = CallbackSchedulerService()

    with patch("app.services.callback_scheduler_service.datetime") as mock_dt:
        mock_dt.utcnow.return_value = now_dt.replace(tzinfo=None)
        svc._dispatch_and_advance(db, schedule)

    assert schedule.status != "executed"
    # Must strictly advance — never equal to (or before) the instant that was
    # just found non-compliant, otherwise the poller re-processes it forever.
    assert schedule.scheduled_at > now_dt
    assert (schedule.scheduled_at - now_dt) >= timedelta(
        days=_MAX_BUSINESS_HOURS_LOOKAHEAD_DAYS
    )
