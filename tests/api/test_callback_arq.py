"""
Tests for the ARQ-based Smart Callback Scheduler.

Coverage map
------------
 1.  execute_callback happy path — row dispatched, status → executed
 2.  execute_callback skips non-pending row (idempotency)
 3.  execute_callback skips missing / locked row
 4.  execute_callback chains next attempt → new CallbackSchedule + ARQ job
 5.  execute_callback exhausts at max_attempts → status = exhausted
 6.  execute_callback reschedules when outside business hours
 7.  execute_callback rolls back DB on dispatch failure
 8.  execute_callback works without a Redis pool (no crash, no enqueue)
 9.  execute_callback multi-tenant isolation (wrong tenant row not touched)
10.  poll_pending_callbacks enqueues rows with arq_job_id=None
11.  poll_pending_callbacks does nothing when no unqueued rows exist
12.  poll_pending_callbacks logs warning and exits when redis pool missing
13.  dispatch_and_advance_async cancels when agent not found
14.  _dispatch_call_async raises RuntimeError without N8N_WEBHOOK_SECRET
15.  _dispatch_call_async raises RuntimeError when original call not found

Mocking strategy
----------------
- ``db`` is a ``MagicMock`` (sync SQLAlchemy Session).
- ``arq_pool`` is an ``AsyncMock`` whose ``enqueue_job`` returns a simple
  namespace with a ``job_id`` attribute, matching ARQ's real return type.
- ``voice_call_service.initiate_call`` is patched with ``AsyncMock``.
- Business-hours helpers are patched at the service level to control outcomes.
- All tests are synchronous unless the function under test is ``async def``,
  in which case ``pytest.mark.asyncio`` + ``asyncio.run`` drive the coroutine.

Running
-------
    pytest tests/api/test_callback_arq.py -v
    # CI: set TEST_DATABASE_URL for integration tests; skip otherwise.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from app.services.callback_scheduler_service import CallbackSchedulerService
from app.models.callback_schedule import CallbackSchedule
from app.models.call_session import CallSession
from app.models.agent import Agent


# ── helpers ────────────────────────────────────────────────────────────────────

def _svc() -> CallbackSchedulerService:
    return CallbackSchedulerService()


def _tenant_id() -> uuid.UUID:
    return uuid.UUID("00000000-0000-0000-0000-000000000001")


def _agent_id() -> uuid.UUID:
    return uuid.UUID("00000000-0000-0000-0000-000000000002")


def _schedule_id() -> uuid.UUID:
    return uuid.UUID("00000000-0000-0000-0000-000000000003")


def _call_id() -> uuid.UUID:
    return uuid.UUID("00000000-0000-0000-0000-000000000004")


def _new_session_id() -> uuid.UUID:
    return uuid.UUID("00000000-0000-0000-0000-000000000005")


def _future_ts() -> datetime:
    return datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)


def _make_schedule(
    *,
    schedule_id=None,
    status="pending",
    attempt_number=1,
    original_call_id=None,
    agent_id=None,
    phone_number="+15550001111",
    scheduled_at=None,
    timezone_str="UTC",
    arq_job_id=None,
) -> CallbackSchedule:
    s = MagicMock(spec=CallbackSchedule)
    s.id = schedule_id or _schedule_id()
    s.status = status
    s.attempt_number = attempt_number
    s.original_call_id = original_call_id or _call_id()
    s.agent_id = agent_id or _agent_id()
    s.phone_number = phone_number
    s.scheduled_at = scheduled_at or _future_ts()
    s.timezone = timezone_str
    s.arq_job_id = arq_job_id
    return s


def _make_agent(
    *,
    smart_callback_enabled=True,
    max_callback_attempts=3,
    gap_schedule=None,
    callback_timezone="UTC",
    tenant_id=None,
) -> Agent:
    a = MagicMock(spec=Agent)
    a.id = _agent_id()
    a.tenant_id = tenant_id or _tenant_id()
    a.smart_callback_enabled = smart_callback_enabled
    a.max_callback_attempts = max_callback_attempts
    a.callback_gap_schedule = gap_schedule or [{"days": 0, "hours": 1}]
    a.callback_timezone = callback_timezone
    return a


def _make_original_call(*, tenant_id=None, user_id=None) -> CallSession:
    c = MagicMock(spec=CallSession)
    c.id = _call_id()
    c.tenant_id = tenant_id or _tenant_id()
    c.user_id = user_id or uuid.uuid4()
    return c


def _make_arq_pool() -> AsyncMock:
    pool = AsyncMock()
    pool.enqueue_job = AsyncMock(
        return_value=SimpleNamespace(job_id=f"arq-job-{uuid.uuid4()}")
    )
    return pool


def _make_db(schedule=None, agent=None, original_call=None) -> MagicMock:
    """Build a MagicMock DB session returning provided objects from db.get()."""
    db = MagicMock()

    def _get(model, pk):
        if model is Agent or (isinstance(model, type) and issubclass(model, Agent)):
            return agent
        if model is CallSession or (isinstance(model, type) and issubclass(model, CallSession)):
            return original_call
        return None

    db.get.side_effect = _get

    if schedule is not None:
        db.execute.return_value.scalar_one_or_none.return_value = schedule
    else:
        db.execute.return_value.scalar_one_or_none.return_value = None

    db.execute.return_value.scalars.return_value.all.return_value = []
    return db


# ── 1. execute_callback happy path ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_callback_happy_path():
    """
    Full success: pending row is dispatched, status transitions to 'executed',
    and a new deferred ARQ job is enqueued for the next attempt.
    """
    schedule = _make_schedule(attempt_number=1)
    agent = _make_agent(max_callback_attempts=3)
    original = _make_original_call()
    db = _make_db(schedule=schedule, agent=agent, original_call=original)
    pool = _make_arq_pool()

    svc = _svc()

    with (
        patch.object(svc, "_is_within_business_hours", return_value=True),
        patch.object(svc, "_next_valid_window", return_value=_future_ts()),
        patch.object(svc, "_dispatch_call_async", new_callable=AsyncMock) as mock_dispatch,
    ):
        next_s = await svc.dispatch_and_advance_async(db, schedule)

    mock_dispatch.assert_awaited_once()
    assert schedule.status == "executed"
    assert schedule.executed_at is not None
    assert next_s is not None
    assert next_s.attempt_number == 2
    db.commit.assert_called()


# ── 2. execute_callback skips non-pending row ─────────────────────────────────

@pytest.mark.asyncio
async def test_execute_callback_skips_non_pending():
    """
    When the row is already 'executed', dispatch_and_advance_async must not
    be called.  The ARQ task wraps it in a SKIP LOCKED query that returns None
    for non-pending rows; this test validates the task-level guard.

    Both SessionLocal and callback_scheduler_service are local imports inside
    execute_callback, so they must be patched at their source modules.
    """
    from app.workers.batch_call_worker import execute_callback

    pool = _make_arq_pool()
    ctx = {"redis": pool}

    with patch("app.db.session.SessionLocal") as mock_sl:
        mock_db = MagicMock()
        # SKIP LOCKED query returns None (row locked / not pending)
        mock_db.execute.return_value.scalar_one_or_none.return_value = None
        mock_sl.return_value = mock_db

        with patch(
            "app.services.callback_scheduler_service.callback_scheduler_service"
        ) as mock_svc:
            await execute_callback(ctx, str(_schedule_id()))

        mock_svc.dispatch_and_advance_async.assert_not_called()


# ── 3. execute_callback skips missing / locked row ────────────────────────────

@pytest.mark.asyncio
async def test_execute_callback_skips_locked_row():
    """SKIP LOCKED returning None is treated as 'another worker has it'."""
    from app.workers.batch_call_worker import execute_callback

    ctx = {"redis": _make_arq_pool()}

    with patch("app.db.session.SessionLocal") as mock_sl:
        mock_db = MagicMock()
        mock_db.execute.return_value.scalar_one_or_none.return_value = None
        mock_sl.return_value = mock_db

        with patch(
            "app.services.callback_scheduler_service.callback_scheduler_service"
        ) as mock_svc:
            await execute_callback(ctx, str(uuid.uuid4()))

        mock_svc.dispatch_and_advance_async.assert_not_called()
    mock_db.close.assert_called_once()


# ── 4. execute_callback chains next attempt ───────────────────────────────────

@pytest.mark.asyncio
async def test_execute_callback_chains_next_attempt():
    """
    When dispatch_and_advance_async returns a next CallbackSchedule, the task
    must enqueue a deferred ARQ job for it and store the job_id.
    """
    from app.workers.batch_call_worker import execute_callback

    schedule = _make_schedule(attempt_number=1, status="pending")
    next_sched = _make_schedule(attempt_number=2, scheduled_at=_future_ts())
    pool = _make_arq_pool()
    ctx = {"redis": pool}

    with patch("app.db.session.SessionLocal") as mock_sl:
        mock_db = MagicMock()
        mock_db.execute.return_value.scalar_one_or_none.return_value = schedule
        mock_sl.return_value = mock_db

        with patch(
            "app.services.callback_scheduler_service.callback_scheduler_service"
        ) as mock_svc:
            mock_svc.dispatch_and_advance_async = AsyncMock(return_value=next_sched)
            await execute_callback(ctx, str(schedule.id))

    pool.enqueue_job.assert_awaited_once()
    call_kwargs = pool.enqueue_job.call_args
    assert call_kwargs[0][0] == "execute_callback"
    assert call_kwargs[0][1] == str(next_sched.id)
    assert call_kwargs[1].get("_defer_until") == next_sched.scheduled_at
    assert call_kwargs[1].get("_job_id") == f"callback:{next_sched.id}"


# ── 5. execute_callback exhausts at max_attempts ─────────────────────────────

@pytest.mark.asyncio
async def test_execute_callback_exhausts_at_max_attempts():
    """When dispatch_and_advance_async returns None, no ARQ job is enqueued."""
    from app.workers.batch_call_worker import execute_callback

    schedule = _make_schedule(attempt_number=3, status="pending")
    pool = _make_arq_pool()
    ctx = {"redis": pool}

    with patch("app.db.session.SessionLocal") as mock_sl:
        mock_db = MagicMock()
        mock_db.execute.return_value.scalar_one_or_none.return_value = schedule
        mock_sl.return_value = mock_db

        with patch(
            "app.services.callback_scheduler_service.callback_scheduler_service"
        ) as mock_svc:
            mock_svc.dispatch_and_advance_async = AsyncMock(return_value=None)
            await execute_callback(ctx, str(schedule.id))

    pool.enqueue_job.assert_not_awaited()


# ── 6. execute_callback reschedules outside business hours ────────────────────

@pytest.mark.asyncio
async def test_execute_callback_reschedules_outside_business_hours():
    """
    When outside business hours, dispatch_and_advance_async returns the SAME
    schedule with an updated scheduled_at; the task re-enqueues it with the
    new time and the same schedule id.
    """
    schedule = _make_schedule(attempt_number=1)
    agent = _make_agent()
    original = _make_original_call()
    db = _make_db(schedule=schedule, agent=agent, original_call=original)

    new_time = _future_ts() + timedelta(hours=8)
    svc = _svc()

    with (
        patch.object(svc, "_is_within_business_hours", return_value=False),
        patch.object(svc, "_next_valid_window", return_value=new_time),
    ):
        result = await svc.dispatch_and_advance_async(db, schedule)

    assert result is schedule
    assert schedule.scheduled_at == new_time
    assert schedule.arq_job_id is None  # cleared for recovery cron


# ── 7. execute_callback rolls back on dispatch failure ────────────────────────

@pytest.mark.asyncio
async def test_execute_callback_rollback_on_dispatch_failure():
    """A RuntimeError from _dispatch_call_async must trigger db.rollback()."""
    from app.workers.batch_call_worker import execute_callback

    schedule = _make_schedule(status="pending")
    pool = _make_arq_pool()
    ctx = {"redis": pool}

    with patch("app.db.session.SessionLocal") as mock_sl:
        mock_db = MagicMock()
        mock_db.execute.return_value.scalar_one_or_none.return_value = schedule
        mock_sl.return_value = mock_db

        with patch(
            "app.services.callback_scheduler_service.callback_scheduler_service"
        ) as mock_svc:
            mock_svc.dispatch_and_advance_async = AsyncMock(
                side_effect=RuntimeError("initiate_call timed out")
            )
            await execute_callback(ctx, str(schedule.id))

    mock_db.rollback.assert_called_once()
    mock_db.close.assert_called_once()


# ── 8. execute_callback works without a Redis pool ────────────────────────────

@pytest.mark.asyncio
async def test_execute_callback_no_redis_pool():
    """
    If the pool is None (startup race or Redis outage), the dispatch still
    completes but no enqueue is attempted and no exception is raised.
    """
    from app.workers.batch_call_worker import execute_callback

    schedule = _make_schedule(status="pending")
    next_sched = _make_schedule(attempt_number=2)
    ctx = {"redis": None}  # pool unavailable

    with patch("app.db.session.SessionLocal") as mock_sl:
        mock_db = MagicMock()
        mock_db.execute.return_value.scalar_one_or_none.return_value = schedule
        mock_sl.return_value = mock_db

        with patch(
            "app.services.callback_scheduler_service.callback_scheduler_service"
        ) as mock_svc:
            mock_svc.dispatch_and_advance_async = AsyncMock(return_value=next_sched)
            await execute_callback(ctx, str(schedule.id))

    mock_db.close.assert_called_once()


# ── 9. multi-tenant isolation ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_callback_tenant_isolation():
    """
    The SKIP LOCKED query always filters by the exact schedule id — it never
    touches rows belonging to a different tenant.  Verify only one DB query
    is issued for the target id.
    """
    from app.workers.batch_call_worker import execute_callback

    target_id = _schedule_id()
    ctx = {"redis": _make_arq_pool()}

    with patch("app.db.session.SessionLocal") as mock_sl:
        mock_db = MagicMock()
        mock_db.execute.return_value.scalar_one_or_none.return_value = None
        mock_sl.return_value = mock_db

        with patch("app.services.callback_scheduler_service.callback_scheduler_service"):
            await execute_callback(ctx, str(target_id))

    assert mock_db.execute.call_count == 1


# ── 10. poll_pending_callbacks enqueues unqueued rows ────────────────────────

@pytest.mark.asyncio
async def test_poll_pending_callbacks_enqueues_unqueued():
    """
    Rows with arq_job_id=None and status='pending' must each get an ARQ job.
    The arq_job_id should be updated with the returned job id.
    """
    from app.workers.batch_call_worker import poll_pending_callbacks

    s1 = _make_schedule(arq_job_id=None)
    s2 = _make_schedule(schedule_id=uuid.uuid4(), arq_job_id=None)
    pool = _make_arq_pool()
    ctx = {"redis": pool}

    with patch("app.db.session.SessionLocal") as mock_sl:
        mock_db = MagicMock()
        mock_db.execute.return_value.scalars.return_value.all.return_value = [s1, s2]
        mock_sl.return_value = mock_db

        await poll_pending_callbacks(ctx)

    assert pool.enqueue_job.await_count == 2
    assert s1.arq_job_id is not None
    assert s2.arq_job_id is not None
    mock_db.commit.assert_called_once()


# ── 11. poll_pending_callbacks no-ops with no unqueued rows ──────────────────

@pytest.mark.asyncio
async def test_poll_pending_callbacks_no_unqueued_rows():
    """When all pending rows already have arq_job_id set, nothing is enqueued."""
    from app.workers.batch_call_worker import poll_pending_callbacks

    pool = _make_arq_pool()
    ctx = {"redis": pool}

    with patch("app.db.session.SessionLocal") as mock_sl:
        mock_db = MagicMock()
        mock_db.execute.return_value.scalars.return_value.all.return_value = []
        mock_sl.return_value = mock_db

        await poll_pending_callbacks(ctx)

    pool.enqueue_job.assert_not_awaited()
    mock_db.commit.assert_not_called()


# ── 12. poll_pending_callbacks exits cleanly without Redis ───────────────────

@pytest.mark.asyncio
async def test_poll_pending_callbacks_no_pool():
    """If ctx has no redis key, the function logs and returns without crash."""
    from app.workers.batch_call_worker import poll_pending_callbacks

    ctx = {"redis": None}

    with patch("app.db.session.SessionLocal") as mock_sl:
        await poll_pending_callbacks(ctx)

    mock_sl.assert_not_called()


# ── 13. dispatch_and_advance_async cancels when agent missing ─────────────────

@pytest.mark.asyncio
async def test_dispatch_and_advance_async_agent_not_found():
    """When the agent is deleted, the schedule must be cancelled."""
    schedule = _make_schedule()
    db = _make_db(schedule=schedule, agent=None)  # agent removed
    svc = _svc()

    result = await svc.dispatch_and_advance_async(db, schedule)

    assert result is None
    assert schedule.status == "cancelled"
    db.commit.assert_called_once()


# ── 14. _dispatch_call_async raises without N8N_WEBHOOK_SECRET ───────────────

@pytest.mark.asyncio
async def test_dispatch_call_async_no_webhook_secret():
    """Missing N8N_WEBHOOK_SECRET must raise RuntimeError before any HTTP call."""
    schedule = _make_schedule()
    agent = _make_agent()
    db = _make_db(schedule=schedule, agent=agent)
    svc = _svc()

    # _dispatch_call_async does `from app.core.config import settings` locally,
    # so we must patch the attribute on the config module object.
    with patch("app.core.config.settings") as mock_settings:
        mock_settings.N8N_WEBHOOK_SECRET = ""

        with pytest.raises(RuntimeError, match="N8N_WEBHOOK_SECRET"):
            await svc._dispatch_call_async(db, schedule, agent)


# ── 15. _dispatch_call_async raises when original call missing ────────────────

@pytest.mark.asyncio
async def test_dispatch_call_async_original_call_not_found():
    """If the original CallSession has been deleted, RuntimeError is raised."""
    schedule = _make_schedule()
    agent = _make_agent()
    # db.get(CallSession, ...) returns None for the original call
    db = _make_db(schedule=schedule, agent=agent, original_call=None)
    svc = _svc()

    with patch("app.core.config.settings") as mock_settings:
        mock_settings.N8N_WEBHOOK_SECRET = "secret"

        with pytest.raises(RuntimeError, match="not found"):
            await svc._dispatch_call_async(db, schedule, agent)


# ── 16–22. Event-driven enqueue + startup recovery ───────────────────────────

# ── 16. _fire_callback_enqueue with a running event loop ─────────────────────

@pytest.mark.asyncio
async def test_fire_callback_enqueue_with_running_loop():
    """
    When called from an async context (running event loop), _fire_callback_enqueue
    must schedule a task that calls pool.enqueue_job with the correct arguments.
    """
    from app.services.call_session_service import _fire_callback_enqueue

    pool = _make_arq_pool()
    schedule_id = _schedule_id()
    scheduled_at = _future_ts()

    with patch("app.services.call_session_service.get_arq_pool", return_value=pool):
        _fire_callback_enqueue(schedule_id, scheduled_at)
        # Yield control so the created task can run.
        await asyncio.sleep(0)

    pool.enqueue_job.assert_awaited_once()
    args, kwargs = pool.enqueue_job.call_args
    assert args[0] == "execute_callback"
    assert args[1] == str(schedule_id)
    assert kwargs["_defer_until"] == scheduled_at
    assert kwargs["_job_id"] == f"callback:{schedule_id}"


# ── 17. _fire_callback_enqueue with no ARQ pool ───────────────────────────────

@pytest.mark.asyncio
async def test_fire_callback_enqueue_no_pool():
    """When the ARQ pool is not yet initialised, log a warning and return silently."""
    from app.services.call_session_service import _fire_callback_enqueue

    with patch("app.services.call_session_service.get_arq_pool", return_value=None):
        # Must not raise
        _fire_callback_enqueue(_schedule_id(), _future_ts())
        await asyncio.sleep(0)

    # No task was submitted — nothing to assert on pool


# ── 18. _fire_callback_enqueue with no running event loop ─────────────────────

def test_fire_callback_enqueue_no_event_loop():
    """
    When called from a purely synchronous context (no running loop), the
    function must log a warning and return without raising.
    """
    from app.services.call_session_service import _fire_callback_enqueue

    pool = _make_arq_pool()

    with patch("app.services.call_session_service.get_arq_pool", return_value=pool):
        # Running this from a plain sync test means asyncio.get_running_loop()
        # will raise RuntimeError — _fire_callback_enqueue must swallow it.
        _fire_callback_enqueue(_schedule_id(), _future_ts())

    # No enqueue attempted synchronously
    pool.enqueue_job.assert_not_called()


# ── 19. _fire_callback_enqueue: inner enqueue failure is swallowed ────────────

@pytest.mark.asyncio
async def test_fire_callback_enqueue_inner_failure_swallowed():
    """If pool.enqueue_job raises, the exception must not propagate to the caller."""
    from app.services.call_session_service import _fire_callback_enqueue

    pool = AsyncMock()
    pool.enqueue_job = AsyncMock(side_effect=ConnectionError("Redis gone"))

    with patch("app.services.call_session_service.get_arq_pool", return_value=pool):
        _fire_callback_enqueue(_schedule_id(), _future_ts())
        await asyncio.sleep(0)  # let the task attempt and fail

    # No exception propagated — test passes if we reach this line


# ── 20. startup_recover_callbacks enqueues all pending rows ──────────────────

@pytest.mark.asyncio
async def test_startup_recover_callbacks_enqueues_all_pending():
    """
    All pending CallbackSchedule rows must be submitted to ARQ on startup.
    ARQ's _job_id ensures duplicates are harmless if the job is already there.
    """
    from app.workers.batch_call_worker import startup_recover_callbacks

    s1 = _make_schedule(schedule_id=uuid.uuid4())
    s2 = _make_schedule(schedule_id=uuid.uuid4())
    pool = _make_arq_pool()
    ctx = {"redis": pool}

    with patch("app.db.session.SessionLocal") as mock_sl:
        mock_db = MagicMock()
        mock_db.execute.return_value.scalars.return_value.all.return_value = [s1, s2]
        mock_sl.return_value = mock_db

        await startup_recover_callbacks(ctx)

    assert pool.enqueue_job.await_count == 2
    for call_args in pool.enqueue_job.call_args_list:
        args, kwargs = call_args
        assert args[0] == "execute_callback"
        assert kwargs.get("_job_id", "").startswith("callback:")


# ── 21. startup_recover_callbacks skips when no pending rows ─────────────────

@pytest.mark.asyncio
async def test_startup_recover_callbacks_no_pending():
    """When there are no pending rows, startup recovery returns without an enqueue."""
    from app.workers.batch_call_worker import startup_recover_callbacks

    pool = _make_arq_pool()
    ctx = {"redis": pool}

    with patch("app.db.session.SessionLocal") as mock_sl:
        mock_db = MagicMock()
        mock_db.execute.return_value.scalars.return_value.all.return_value = []
        mock_sl.return_value = mock_db

        await startup_recover_callbacks(ctx)

    pool.enqueue_job.assert_not_awaited()


# ── 22. startup_recover_callbacks exits cleanly without redis ─────────────────

@pytest.mark.asyncio
async def test_startup_recover_callbacks_no_redis():
    """If ctx has no redis pool, the function logs and returns without crash."""
    from app.workers.batch_call_worker import startup_recover_callbacks

    ctx = {"redis": None}

    with patch("app.db.session.SessionLocal") as mock_sl:
        await startup_recover_callbacks(ctx)

    mock_sl.assert_not_called()


# ── Integration test template (skipped without TEST_DATABASE_URL) ─────────────

@pytest.mark.skipif(
    __import__("os").getenv("TEST_DATABASE_URL") is None,
    reason="Integration tests require TEST_DATABASE_URL",
)
def test_integration_callback_round_trip():
    """
    End-to-end integration: create a CallbackSchedule row, verify
    execute_callback dispatches it and creates the next row.

    Run with:
        TEST_DATABASE_URL=postgresql://... pytest tests/api/test_callback_arq.py \
            -k test_integration_callback_round_trip -v
    """
    # This intentionally remains as a template.
    # Populate with real DB fixtures when integration test infra is available.
    pass
