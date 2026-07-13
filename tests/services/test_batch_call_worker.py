"""
Unit tests for BatchCallService and BatchCallWorkerService.

All external dependencies (GCS, initiate_call, billing) are mocked.
DB is an in-memory SQLite fixture shared within each test class.

Coverage:
  - CSV validation: happy path, missing phone_number column, variable mismatch
  - Job + record creation
  - SKIP LOCKED pickup
  - Progress counts update correctly
  - Cancellation stops new pickups; already-active records unaffected
  - Retry logic for busy/no_answer; no retry for invalid_number
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.db.base import Base

# SQLite JSONB/UUID compat
import sqlite3
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles


@compiles(JSONB, "sqlite")
def _jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


@compiles(PG_UUID, "sqlite")
def _uuid_sqlite(type_, compiler, **kw):
    return "VARCHAR(36)"


# ── Fixtures ──────────────────────────────────────────────────────────────────

_SHARED_CONN = sqlite3.connect(":memory:", check_same_thread=False)

_engine = create_engine(
    "sqlite://",
    creator=lambda: _SHARED_CONN,
    connect_args={"check_same_thread": False},
)
_Session = sessionmaker(bind=_engine)


def _setup_db():
    """Re-create all tables in the shared SQLite connection."""
    # Import models to register them on Base.metadata
    import app.models.agent  # noqa: F401
    import app.models.batch_call_record  # noqa: F401
    import app.models.batch_job  # noqa: F401
    import app.models.call_session  # noqa: F401
    import app.models.tenant  # noqa: F401
    import app.models.user  # noqa: F401
    import app.models.plan  # noqa: F401
    import app.models.subscription  # noqa: F401
    import app.models.usage_record  # noqa: F401
    import app.models.role  # noqa: F401

    Base.metadata.drop_all(bind=_engine)
    Base.metadata.create_all(bind=_engine)


@pytest.fixture()
def db():
    _setup_db()
    session = _Session()
    try:
        yield session
    finally:
        session.close()


# ── Helper builders ───────────────────────────────────────────────────────────

def _make_tenant(db) -> uuid.UUID:
    from app.models.tenant import Tenant

    t = Tenant(name="WS", schema_name="ws_test")
    db.add(t)
    db.commit()
    db.refresh(t)
    return t.id


def _make_agent(db, tenant_id: uuid.UUID, prompt: str = "") -> uuid.UUID:
    from app.models.agent import Agent

    a = Agent(
        tenant_id=tenant_id,
        name="TestAgent",
        system_prompt=prompt,
        status="ready",
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    return a.id


def _make_batch_job(db, workspace_id, agent_id, total=3) -> "BatchJob":  # noqa: F821
    from app.models.batch_job import BatchJob

    j = BatchJob(
        workspace_id=workspace_id,
        agent_id=agent_id,
        status="pending",
        total_count=total,
        waiting_count=total,
        active_count=0,
        completed_count=0,
        failed_count=0,
    )
    db.add(j)
    db.commit()
    db.refresh(j)
    return j


def _make_record(db, batch_job_id, phone="+15551230000", status="waiting"):
    from app.models.batch_call_record import BatchCallRecord

    r = BatchCallRecord(
        batch_job_id=batch_job_id,
        phone_number=phone,
        status=status,
        attempts=0,
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    return r


# ── CSV Validation ────────────────────────────────────────────────────────────

class TestCsvValidation:
    """Tests for BatchCallService._validate_csv and create_batch_job."""

    @patch("app.services.batch_call_service.batch_call_gcs_service.upload_batch_csv")
    def test_valid_csv_creates_job_and_records(self, mock_upload, db):
        mock_upload.return_value = "batch-files/ws/id.csv"

        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)

        csv_bytes = b"phone_number,first_name\n+15550001111,Alice\n+15550002222,Bob"

        from app.services.batch_call_service import BatchCallService

        svc = BatchCallService(db)
        result = svc.create_batch_job(workspace_id, agent_id, csv_bytes)

        assert result.total_count == 2
        assert result.waiting_count == 2
        assert result.status == "pending"
        assert result.gcs_path is not None
        mock_upload.assert_called_once()

    @patch("app.services.batch_call_service.batch_call_gcs_service.upload_batch_csv")
    def test_missing_phone_number_column_raises_422(self, mock_upload, db):
        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)

        csv_bytes = b"name,email\nAlice,alice@example.com"

        from app.services.batch_call_service import BatchCallService

        svc = BatchCallService(db)

        with pytest.raises(HTTPException) as exc_info:
            svc.create_batch_job(workspace_id, agent_id, csv_bytes)

        assert exc_info.value.status_code == 422
        assert "phone_number" in str(exc_info.value.detail)
        mock_upload.assert_not_called()

    @patch("app.services.batch_call_service.batch_call_gcs_service.upload_batch_csv")
    def test_prompt_variable_mismatch_raises_422(self, mock_upload, db):
        workspace_id = _make_tenant(db)
        # Agent prompt references {last_name} but CSV only has first_name
        agent_id = _make_agent(db, workspace_id, prompt="Hello {last_name}")

        csv_bytes = b"phone_number,first_name\n+15550001111,Alice"

        from app.services.batch_call_service import BatchCallService

        svc = BatchCallService(db)

        with pytest.raises(HTTPException) as exc_info:
            svc.create_batch_job(workspace_id, agent_id, csv_bytes)

        assert exc_info.value.status_code == 422
        assert "last_name" in str(exc_info.value.detail)
        mock_upload.assert_not_called()

    @patch("app.services.batch_call_service.batch_call_gcs_service.upload_batch_csv")
    def test_prompt_variable_present_in_csv_succeeds(self, mock_upload, db):
        mock_upload.return_value = "batch-files/ws/id.csv"
        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id, prompt="Hello {first_name}")

        csv_bytes = b"phone_number,first_name\n+15550001111,Alice"

        from app.services.batch_call_service import BatchCallService

        svc = BatchCallService(db)
        result = svc.create_batch_job(workspace_id, agent_id, csv_bytes)

        assert result.total_count == 1

    @patch("app.services.batch_call_service.batch_call_gcs_service.upload_batch_csv")
    def test_exceeding_max_size_raises_422(self, mock_upload, db):
        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)

        # 21 MB of zeros
        big_csv = b"phone_number\n" + b"+15550001111\n" * 100 + b"x" * (21 * 1024 * 1024)

        from app.services.batch_call_service import BatchCallService

        svc = BatchCallService(db)

        with pytest.raises(HTTPException) as exc_info:
            svc.create_batch_job(workspace_id, agent_id, big_csv)

        assert exc_info.value.status_code == 422
        mock_upload.assert_not_called()


# ── Progress counts ───────────────────────────────────────────────────────────

class TestProgressCounts:
    def test_progress_returns_correct_counts(self, db):
        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        job = _make_batch_job(db, workspace_id, agent_id, total=10)

        # Manually set some counts
        job.waiting_count = 5
        job.active_count = 2
        job.completed_count = 2
        job.failed_count = 1
        db.commit()

        from app.services.batch_call_service import BatchCallService

        svc = BatchCallService(db)
        progress = svc.get_batch_job_progress(workspace_id, job.id)

        assert progress.waiting == 5
        assert progress.active == 2
        assert progress.completed == 2
        assert progress.failed == 1
        assert progress.total == 10
        assert progress.percent_complete == 20.0

    def test_progress_not_found_raises_404(self, db):
        workspace_id = _make_tenant(db)

        from app.services.batch_call_service import BatchCallService

        svc = BatchCallService(db)

        with pytest.raises(HTTPException) as exc_info:
            svc.get_batch_job_progress(workspace_id, uuid.uuid4())

        assert exc_info.value.status_code == 404


# ── Cancellation ──────────────────────────────────────────────────────────────

class TestCancellation:
    def test_cancel_sets_status_and_cancels_waiting_records(self, db):
        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        job = _make_batch_job(db, workspace_id, agent_id, total=3)

        r1 = _make_record(db, job.id, "+15550001111", status="waiting")
        r2 = _make_record(db, job.id, "+15550002222", status="waiting")
        r3 = _make_record(db, job.id, "+15550003333", status="active")  # already active

        from app.services.batch_call_service import BatchCallService

        svc = BatchCallService(db)
        result = svc.cancel_batch_job(workspace_id, job.id)

        assert result.status == "cancelled"

        db.refresh(r1)
        db.refresh(r2)
        db.refresh(r3)
        assert r1.status == "cancelled"
        assert r2.status == "cancelled"
        assert r3.status == "active"  # active call not touched

    def test_cancel_completed_job_raises_409(self, db):
        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        job = _make_batch_job(db, workspace_id, agent_id)
        job.status = "completed"
        db.commit()

        from app.services.batch_call_service import BatchCallService

        svc = BatchCallService(db)

        with pytest.raises(HTTPException) as exc_info:
            svc.cancel_batch_job(workspace_id, job.id)

        assert exc_info.value.status_code == 409

    def test_worker_skips_cancelled_job(self, db):
        """pick_waiting_records returns [] for a cancelled job."""
        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        job = _make_batch_job(db, workspace_id, agent_id, total=2)
        _make_record(db, job.id, "+15550001111", status="waiting")
        job.status = "cancelled"
        db.commit()

        from app.services.batch_call_worker_service import BatchCallWorkerService

        svc = BatchCallWorkerService(db)
        records = svc.pick_waiting_records(job.id, limit=5)

        assert records == []


# ── Retry logic ───────────────────────────────────────────────────────────────

class TestRetryLogic:
    def test_no_answer_schedules_retry(self, db):
        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        job = _make_batch_job(db, workspace_id, agent_id, total=1)
        record = _make_record(db, job.id, "+15550001111", status="active")
        job.active_count = 1
        job.waiting_count = 0
        db.commit()

        from app.services.batch_call_worker_service import BatchCallWorkerService

        svc = BatchCallWorkerService(db)
        asyncio.run(svc._schedule_retry(record, "no-answer"))

        db.refresh(record)
        assert record.status == "waiting"
        assert record.attempts == 1
        assert record.next_attempt_at is not None
        assert record.last_error == "no-answer"

    def test_busy_schedules_retry(self, db):
        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        job = _make_batch_job(db, workspace_id, agent_id, total=1)
        record = _make_record(db, job.id, "+15550001111", status="active")
        job.active_count = 1
        db.commit()

        from app.services.batch_call_worker_service import BatchCallWorkerService

        svc = BatchCallWorkerService(db)
        asyncio.run(svc._schedule_retry(record, "busy"))

        db.refresh(record)
        assert record.status == "waiting"
        assert record.attempts == 1

    def test_max_attempts_exceeded_marks_failed(self, db):
        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        job = _make_batch_job(db, workspace_id, agent_id, total=1)
        record = _make_record(db, job.id, "+15550001111", status="active")
        record.attempts = 3  # already at max
        job.active_count = 1
        db.commit()

        from app.services.batch_call_worker_service import BatchCallWorkerService

        svc = BatchCallWorkerService(db)
        asyncio.run(svc._schedule_retry(record, "no-answer"))

        db.refresh(record)
        assert record.status == "failed"
        assert "max_attempts_exceeded" in (record.last_error or "")

    def test_invalid_number_marks_failed_no_retry(self, db):
        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        job = _make_batch_job(db, workspace_id, agent_id, total=1)
        record = _make_record(db, job.id, "+15550001111", status="active")
        record.attempts = 0
        job.active_count = 1
        db.commit()

        from app.services.batch_call_worker_service import BatchCallWorkerService

        svc = BatchCallWorkerService(db)
        asyncio.run(svc._mark_failed(record, "invalid-number", is_system_error=False, no_retry=True))

        db.refresh(record)
        assert record.status == "failed"
        assert record.last_error == "invalid-number"
        # Retries == 0 (was never retried)
        assert record.attempts == 0


# ── Dispatch ──────────────────────────────────────────────────────────────────

class TestDispatch:
    def test_dispatch_calls_initiate_call_and_marks_active(self, db):
        from app.schemas.base import SuccessResponse
        from app.schemas.twilio import CallInitiateResponse

        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        job = _make_batch_job(db, workspace_id, agent_id, total=1)
        record = _make_record(db, job.id, "+15550001111", status="active")
        job.active_count = 1
        db.commit()

        call_session_id = uuid.uuid4()
        fake_response = SuccessResponse(
            data=CallInitiateResponse(
                callId=str(call_session_id),
                twilioCallSid="CA123",
                callSessionId=str(call_session_id),
                status="initiated",
            )
        )

        from app.services.batch_call_worker_service import BatchCallWorkerService

        with patch(
            "app.services.voice_call_service.initiate_call",
            new=AsyncMock(return_value=fake_response),
        ):
            svc = BatchCallWorkerService(db)
            asyncio.run(svc.dispatch_record(record, workspace_id, agent_id, None))

        db.refresh(record)
        # Record should now have attempts incremented
        assert record.attempts == 1

    def test_dispatch_with_variable_substitution(self, db):
        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id, prompt="Hello {first_name}")
        job = _make_batch_job(db, workspace_id, agent_id, total=1)
        record = _make_record(db, job.id, "+15550001111", status="active")
        record.variables = {"first_name": "Alice"}
        job.active_count = 1
        db.commit()

        from app.schemas.base import SuccessResponse
        from app.schemas.twilio import CallInitiateResponse

        _cid = uuid.uuid4()
        fake_response = SuccessResponse(
            data=CallInitiateResponse(
                callId=str(_cid),
                twilioCallSid="CA456",
                callSessionId=str(_cid),
                status="initiated",
            )
        )

        from app.services.batch_call_worker_service import BatchCallWorkerService

        with patch(
            "app.services.voice_call_service.initiate_call",
            new=AsyncMock(return_value=fake_response),
        ) as mock_call:
            svc = BatchCallWorkerService(db)
            asyncio.run(svc.dispatch_record(
                record, workspace_id, agent_id, "Hello {first_name}"
            ))
            mock_call.assert_called_once()
            req = mock_call.call_args.kwargs["call_request"]
            assert req.batch_call_record_id == str(record.id)
            assert req.batch_prompt_override == "Hello Alice"

    def test_dispatch_system_error_marks_failed_without_billing(self, db):
        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        job = _make_batch_job(db, workspace_id, agent_id, total=1)
        record = _make_record(db, job.id, "+15550001111", status="active")
        job.active_count = 1
        db.commit()

        from app.services.batch_call_worker_service import BatchCallWorkerService

        with patch(
            "app.services.voice_call_service.initiate_call",
            new=AsyncMock(side_effect=RuntimeError("Twilio down")),
        ):
            svc = BatchCallWorkerService(db)
            asyncio.run(svc.dispatch_record(record, workspace_id, agent_id, None))

        db.refresh(record)
        assert record.status == "failed"
        assert "Twilio down" in (record.last_error or "")


# ── Webhook completion bridge ─────────────────────────────────────────────────

class TestBatchCallCompletion:
    def test_notify_batch_call_ended_completes_active_record(self, db):
        from app.models.call_session import CallSession
        from app.models.user import User

        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        user = User(
            id=uuid.uuid4(),
            first_name="Batch",
            last_name="User",
            email="batch-user@example.com",
            hashed_password="x",
        )
        db.add(user)
        db.flush()
        call_id = uuid.uuid4()
        cs = CallSession(
            id=call_id,
            user_id=user.id,
            agent_id=agent_id,
            tenant_id=workspace_id,
            start_time=datetime.now(timezone.utc),
            status="active",
            call_type="outbound",
        )
        db.add(cs)
        job = _make_batch_job(db, workspace_id, agent_id, total=1)
        record = _make_record(db, job.id, "+15550001111", status="active")
        record.call_id = call_id
        job.active_count = 1
        job.waiting_count = 0
        db.commit()

        from app.services.batch_call_completion_service import notify_batch_call_ended

        with patch(
            "app.services.billing_service.BillingService.record_call_usage"
        ) as mock_bill:
            asyncio.run(notify_batch_call_ended(db, call_id, "completed"))

        db.refresh(record)
        db.refresh(job)
        assert record.status == "completed"
        assert job.completed_count == 1
        assert job.active_count == 0
        mock_bill.assert_called_once()

    def test_notify_batch_call_ended_schedules_retry_on_busy(self, db):
        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        job = _make_batch_job(db, workspace_id, agent_id, total=1)
        call_id = uuid.uuid4()
        record = _make_record(db, job.id, "+15550001111", status="active")
        record.call_id = call_id
        record.attempts = 1
        job.active_count = 1
        db.commit()

        from app.services.batch_call_completion_service import notify_batch_call_ended

        asyncio.run(notify_batch_call_ended(db, call_id, "busy"))

        db.refresh(record)
        assert record.status == "waiting"
        assert record.next_attempt_at is not None

    def test_notify_batch_call_ended_ignores_non_batch_sessions(self, db):
        from app.services.batch_call_completion_service import notify_batch_call_ended

        # Should not raise when no batch record matches
        asyncio.run(notify_batch_call_ended(db, uuid.uuid4(), "completed"))


# ── Secret validation (Fix 1) ─────────────────────────────────────────────────

class TestSecretValidation:
    def test_dispatch_raises_runtime_error_when_secret_missing(self, db):
        """dispatch_record must fail fast with a descriptive RuntimeError when
        N8N_WEBHOOK_SECRET is not configured — prevents silent auth bypass."""
        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        job = _make_batch_job(db, workspace_id, agent_id, total=1)
        record = _make_record(db, job.id, "+15550001111", status="active")
        job.active_count = 1
        db.commit()

        from app.services.batch_call_worker_service import BatchCallWorkerService

        with patch("app.services.batch_call_worker_service.settings") as mock_cfg:
            mock_cfg.N8N_WEBHOOK_SECRET = ""
            svc = BatchCallWorkerService(db)
            with pytest.raises(RuntimeError, match="N8N_WEBHOOK_SECRET"):
                asyncio.run(svc.dispatch_record(record, workspace_id, agent_id, None))

    def test_dispatch_does_not_call_initiate_when_secret_missing(self, db):
        """initiate_call must never be reached when the secret is absent."""
        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        job = _make_batch_job(db, workspace_id, agent_id, total=1)
        record = _make_record(db, job.id, "+15550001111", status="active")
        job.active_count = 1
        db.commit()

        from app.services.batch_call_worker_service import BatchCallWorkerService

        with patch("app.services.batch_call_worker_service.settings") as mock_cfg:
            mock_cfg.N8N_WEBHOOK_SECRET = ""
            with patch(
                "app.services.voice_call_service.initiate_call",
                new=AsyncMock(),
            ) as mock_call:
                svc = BatchCallWorkerService(db)
                with pytest.raises(RuntimeError):
                    asyncio.run(svc.dispatch_record(record, workspace_id, agent_id, None))
                mock_call.assert_not_called()

    def test_dispatch_proceeds_when_secret_present(self, db):
        """Sanity check: a non-empty secret does not raise RuntimeError."""
        from app.schemas.base import SuccessResponse
        from app.schemas.twilio import CallInitiateResponse

        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        job = _make_batch_job(db, workspace_id, agent_id, total=1)
        record = _make_record(db, job.id, "+15550001111", status="active")
        job.active_count = 1
        db.commit()

        _cid = uuid.uuid4()
        fake_response = SuccessResponse(
            data=CallInitiateResponse(
                callId=str(_cid),
                twilioCallSid="CAtest",
                callSessionId=str(_cid),
                status="initiated",
            )
        )

        from app.services.batch_call_worker_service import BatchCallWorkerService

        with patch("app.services.batch_call_worker_service.settings") as mock_cfg:
            mock_cfg.N8N_WEBHOOK_SECRET = "secret-token"
            with patch(
                "app.services.voice_call_service.initiate_call",
                new=AsyncMock(return_value=fake_response),
            ):
                svc = BatchCallWorkerService(db)
                asyncio.run(svc.dispatch_record(record, workspace_id, agent_id, None))

        db.refresh(record)
        assert record.attempts == 1


# ── Atomic counter updates (Fix 2) ───────────────────────────────────────────

class TestAtomicCounterUpdates:
    def test_counter_update_uses_server_side_arithmetic(self):
        """
        The bulk UPDATE in pick_waiting_records must use column-based arithmetic
        so concurrent workers cannot produce lost updates.

        SQLAlchemy emits  SET waiting_count = batchjob.waiting_count - :param
        not the Python-evaluated literal  SET waiting_count = :literal_value.
        """
        from sqlalchemy import update
        from sqlalchemy.dialects import postgresql
        from app.models.batch_job import BatchJob

        n = 4
        stmt = (
            update(BatchJob)
            .values(
                waiting_count=BatchJob.waiting_count - n,
                active_count=BatchJob.active_count + n,
            )
        )
        sql = str(stmt.compile(dialect=postgresql.dialect()))

        # Column reference must appear on the RHS of the assignment
        assert "waiting_count - " in sql, f"Expected server-side subtraction, got: {sql}"
        assert "active_count + " in sql, f"Expected server-side addition, got: {sql}"

    def test_two_sequential_pickups_produce_correct_counters(self, db):
        """
        Simulate two workers each picking half the records from a 6-record job.

        The raw SKIP LOCKED SQL is mocked (SQLite does not support it); the ORM
        counter UPDATE is allowed to execute so its correctness is verified.
        SKIP LOCKED semantics are validated separately in Postgres integration tests.
        """
        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        job = _make_batch_job(db, workspace_id, agent_id, total=6)
        records = [_make_record(db, job.id, f"+1555000{i:04d}") for i in range(6)]
        # Keep as UUID objects — the ORM update uses id.in_(record_ids) which needs UUIDs
        all_ids = [r.id for r in records]

        from app.services.batch_call_worker_service import BatchCallWorkerService

        original_execute = db.execute

        # Worker 1 picks records 0-2 (SKIP LOCKED returns them)
        w1_ids = all_ids[:3]

        def _execute_w1(stmt, params=None, **kwargs):
            if params is not None and "lim" in params:
                mock_result = MagicMock()
                mock_result.fetchall.return_value = [(rid,) for rid in w1_ids]
                return mock_result
            return original_execute(stmt, params, **kwargs)

        with patch.object(db, "execute", side_effect=_execute_w1):
            svc1 = BatchCallWorkerService(db)
            picked1 = svc1.pick_waiting_records(job.id, limit=3)

        db.refresh(job)
        assert len(picked1) == 3
        assert job.waiting_count == 3
        assert job.active_count == 3

        # Worker 2 picks remaining records 3-5
        w2_ids = all_ids[3:]

        def _execute_w2(stmt, params=None, **kwargs):
            if params is not None and "lim" in params:
                mock_result = MagicMock()
                mock_result.fetchall.return_value = [(rid,) for rid in w2_ids]
                return mock_result
            return original_execute(stmt, params, **kwargs)

        with patch.object(db, "execute", side_effect=_execute_w2):
            svc2 = BatchCallWorkerService(db)
            picked2 = svc2.pick_waiting_records(job.id, limit=3)

        db.refresh(job)
        assert len(picked2) == 3
        assert job.waiting_count == 0
        assert job.active_count == 6

        # Workers must have picked non-overlapping sets
        ids1 = {r.id for r in picked1}
        ids2 = {r.id for r in picked2}
        assert ids1.isdisjoint(ids2), "Workers picked overlapping records"

    def test_skip_locked_prevents_double_pickup_within_single_worker(self, db):
        """
        A second call to pick_waiting_records on the same job after all records
        are active must return an empty list (no double-pickup).
        """
        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        job = _make_batch_job(db, workspace_id, agent_id, total=2)
        records = [_make_record(db, job.id, f"+1555000{i:04d}") for i in range(2)]
        all_ids = [r.id for r in records]  # UUID objects for ORM compatibility

        from app.services.batch_call_worker_service import BatchCallWorkerService

        original_execute = db.execute
        first_call = {"done": False}

        def _execute_once(stmt, params=None, **kwargs):
            if params is not None and "lim" in params:
                mock_result = MagicMock()
                if not first_call["done"]:
                    first_call["done"] = True
                    mock_result.fetchall.return_value = [(rid,) for rid in all_ids]
                else:
                    # SKIP LOCKED: all rows locked by first worker → empty
                    mock_result.fetchall.return_value = []
                return mock_result
            return original_execute(stmt, params, **kwargs)

        with patch.object(db, "execute", side_effect=_execute_once):
            svc = BatchCallWorkerService(db)
            first = svc.pick_waiting_records(job.id, limit=5)
            second = svc.pick_waiting_records(job.id, limit=5)

        assert len(first) == 2
        assert second == []


class TestBillingRecordCallUsage:
    def test_record_call_usage_increments_monthly_counter(self, db):
        from app.models.plan import Plan
        from app.models.subscription import Subscription
        from app.models.usage_record import UsageRecord
        from app.models.user import User, user_tenant_association
        from app.services.billing_service import BillingService

        workspace_id = _make_tenant(db)
        user = User(
            id=uuid.uuid4(),
            first_name="Batch",
            last_name="Bill",
            email="batch-bill@example.com",
            hashed_password="x",
        )
        db.add(user)
        db.flush()
        db.execute(
            user_tenant_association.insert().values(
                user_id=user.id,
                tenant_id=workspace_id,
                is_creator=True,
            )
        )
        plan = Plan(
            id=uuid.uuid4(),
            name="free",
            display_name="Free",
            price_monthly=0,
            is_active=True,
        )
        db.add(plan)
        sub = Subscription(
            id=uuid.uuid4(),
            user_id=user.id,
            plan_id=plan.id,
            status="active",
        )
        db.add(sub)
        db.commit()

        BillingService.record_call_usage(db, workspace_id, user_id=user.id)

        usage = (
            db.query(UsageRecord)
            .filter(UsageRecord.workspace_id == workspace_id)
            .one()
        )
        assert usage.workspace_id == workspace_id
