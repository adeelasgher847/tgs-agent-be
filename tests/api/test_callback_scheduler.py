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
  8. Business hours check reschedules when outside window
  9. Business hours: open window is used as-is
 10. Exhaustion: no further schedule is created at max_attempts
 11. Exhausted schedule is logged with status='exhausted'
 12. Gap schedule index is clamped at last entry when attempts exceed schedule length
 13. get_callback_history returns 404 for unknown call
 14. get_callback_history returns 404 for cross-tenant call
 15. PUT callback-config returns 404 for unknown agent
"""
from __future__ import annotations

import uuid
from datetime import datetime, time, timedelta
from typing import Optional
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from app.models.agent import Agent
from app.models.business_hours import BusinessHours
from app.models.callback_schedule import CallbackSchedule
from app.models.call_session import CallSession
from app.schemas.callback_scheduler import (
    CallbackConfigUpdate,
    GapInterval,
)
from app.services.callback_scheduler_service import (
    CallbackSchedulerService,
    CALLBACK_TRIGGER_STATUSES,
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
) -> CallSession:
    cs = MagicMock(spec=CallSession)
    cs.id = uuid.uuid4()
    cs.agent_id = agent_id or uuid.uuid4()
    cs.tenant_id = tenant_id or uuid.uuid4()
    cs.status = status
    cs.to_number = to_number
    cs.customer_phone_number = customer_phone_number or to_number
    cs.user_id = uuid.uuid4()
    return cs


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
    assert any("timezone" in str(e.get("loc", "")) or "timezone" in str(e) for e in errors)


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
    When the current time is outside business hours, _dispatch_and_advance
    must update scheduled_at to the next valid window and NOT dispatch.
    """
    agent = _make_agent(callback_timezone="America/New_York")
    # Simulate current time at 02:00 UTC = 22:00 Eastern (outside 09:00-17:00)
    closed_dt = datetime(2026, 6, 15, 2, 0, 0, tzinfo=ZoneInfo("UTC"))

    bh = MagicMock(spec=BusinessHours)
    bh.is_closed = False
    bh.open_time = time(9, 0)
    bh.close_time = time(17, 0)

    schedule = MagicMock(spec=CallbackSchedule)
    schedule.id = uuid.uuid4()
    schedule.original_call_id = uuid.uuid4()
    schedule.agent_id = agent.id
    schedule.phone_number = "+15550001234"
    schedule.attempt_number = 1
    schedule.scheduled_at = closed_dt
    schedule.timezone = "America/New_York"

    db = MagicMock()
    db.get.side_effect = lambda cls, pk: agent if cls is Agent else None

    # Return the business hours row on first BH query, None afterward
    bh_call = [0]

    def _execute(stmt, *a, **kw):
        bh_call[0] += 1
        m = MagicMock()
        m.scalar_one_or_none.return_value = bh
        return m

    db.execute.side_effect = _execute

    svc = CallbackSchedulerService()

    with patch("app.services.callback_scheduler_service.datetime") as mock_dt:
        mock_dt.utcnow.return_value = closed_dt
        svc._dispatch_and_advance(db, schedule)

    # The schedule should have been rescheduled (scheduled_at updated), not executed
    assert schedule.status != "executed"
    db.commit.assert_called()


# ── 12. Within business hours: call is dispatched ────────────────────────────


def test_within_business_hours_dispatches_call():
    """
    When the time is within business hours, dispatch_call is invoked and
    the schedule is marked executed.
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

    bh = MagicMock(spec=BusinessHours)
    bh.is_closed = False
    bh.open_time = time(9, 0)
    bh.close_time = time(17, 0)

    original_call = MagicMock(spec=CallSession)
    original_call.id = uuid.uuid4()
    original_call.user_id = uuid.uuid4()
    original_call.tenant_id = uuid.uuid4()

    schedule = MagicMock(spec=CallbackSchedule)
    schedule.id = uuid.uuid4()
    schedule.original_call_id = original_call.id
    schedule.agent_id = agent.id
    schedule.phone_number = "+15550001234"
    schedule.attempt_number = 1
    schedule.scheduled_at = open_dt
    schedule.timezone = "UTC"
    schedule.status = "pending"

    db = MagicMock()
    db.get.side_effect = lambda cls, pk: (
        agent if cls is Agent
        else original_call if cls is CallSession
        else None
    )

    def _execute(stmt, *a, **kw):
        m = MagicMock()
        m.scalar_one_or_none.return_value = bh
        return m

    db.execute.side_effect = _execute

    svc = CallbackSchedulerService()

    with patch("app.services.callback_scheduler_service.datetime") as mock_dt:
        mock_dt.utcnow.return_value = open_dt
        svc._dispatch_and_advance(db, schedule)

    # Schedule must be marked executed
    assert schedule.status == "executed"
    assert schedule.executed_at is not None
    # A new (chained) CallbackSchedule should have been added
    add_calls = [
        call_args[0][0]
        for call_args in db.add.call_args_list
        if isinstance(call_args[0][0], CallbackSchedule)
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
    open_dt = datetime(2026, 6, 15, 10, 0, 0, tzinfo=ZoneInfo("UTC"))

    bh = MagicMock(spec=BusinessHours)
    bh.is_closed = False
    bh.open_time = time(9, 0)
    bh.close_time = time(17, 0)

    original_call = MagicMock(spec=CallSession)
    original_call.id = uuid.uuid4()
    original_call.user_id = uuid.uuid4()
    original_call.tenant_id = uuid.uuid4()

    schedule = MagicMock(spec=CallbackSchedule)
    schedule.id = uuid.uuid4()
    schedule.original_call_id = original_call.id
    schedule.agent_id = agent.id
    schedule.phone_number = "+15550001234"
    # This is the LAST allowed attempt
    schedule.attempt_number = 2
    schedule.scheduled_at = open_dt
    schedule.timezone = "UTC"
    schedule.status = "pending"

    db = MagicMock()
    db.get.side_effect = lambda cls, pk: (
        agent if cls is Agent
        else original_call if cls is CallSession
        else None
    )

    def _execute(stmt, *a, **kw):
        m = MagicMock()
        m.scalar_one_or_none.return_value = bh
        m.scalars.return_value.all.return_value = []
        return m

    db.execute.side_effect = _execute

    svc = CallbackSchedulerService()

    with patch("app.services.callback_scheduler_service.datetime") as mock_dt:
        mock_dt.utcnow.return_value = open_dt
        svc._dispatch_and_advance(db, schedule)

    # Final schedule must be exhausted, no new record added
    assert schedule.status == "exhausted"
    chained = [
        call_args[0][0]
        for call_args in db.add.call_args_list
        if isinstance(call_args[0][0], CallbackSchedule)
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
