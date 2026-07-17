"""
Tests for outbound number reputation monitoring / auto-rotation.

Coverage:
  - BatchCallService.rotate_number_if_flagged: area-code preference, same-country
    fallback, AllNumbersFlaggedError when no clean number exists, no-op when clean.
  - POST /batch-calls: 422 all_numbers_flagged payload + batch.number_rotated audit log.
  - ARQ cron check_all_phone_numbers_reputation: only checks numbers missing/stale (>24h).

DB is an in-memory SQLite fixture (mirrors tests/services/test_batch_call_worker.py).
The reputation provider call is mocked — no real HTTP calls.
"""
from __future__ import annotations

import io
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker

from app.db.base import Base


@compiles(JSONB, "sqlite")
def _jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


@compiles(PG_UUID, "sqlite")
def _uuid_sqlite(type_, compiler, **kw):
    return "VARCHAR(36)"


_SHARED_CONN = sqlite3.connect(":memory:", check_same_thread=False)
_engine = create_engine(
    "sqlite://",
    creator=lambda: _SHARED_CONN,
    connect_args={"check_same_thread": False},
)
_Session = sessionmaker(bind=_engine)


def _setup_db():
    import app.models.agent  # noqa: F401
    import app.models.batch_call_record  # noqa: F401
    import app.models.batch_job  # noqa: F401
    import app.models.phone_number  # noqa: F401
    import app.models.phone_number_reputation  # noqa: F401
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


def _make_agent(db, tenant_id: uuid.UUID) -> uuid.UUID:
    from app.models.agent import Agent

    a = Agent(tenant_id=tenant_id, name="TestAgent", status="ready")
    db.add(a)
    db.commit()
    db.refresh(a)
    return a.id


def _make_phone_number(db, tenant_id, agent_id, number: str):
    from app.models.phone_number import PhoneNumber

    p = PhoneNumber(
        tenant_id=tenant_id,
        phone_number=number,
        status="active",
        provider="twilio",
        assistant_id=agent_id,
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


def _make_extra_number(db, tenant_id, number: str):
    """A pool number not bound to any agent."""
    return _make_phone_number(db, tenant_id, None, number)


def _make_reputation(db, phone_number_id, *, spam_flagged: bool, score: int = 30, checked_at=None):
    from app.models.phone_number_reputation import PhoneNumberReputation

    r = PhoneNumberReputation(
        phone_number_id=phone_number_id,
        reputation_score=score,
        spam_flagged=spam_flagged,
        last_checked_at=checked_at or datetime.now(timezone.utc),
        checked_by="mock",
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    return r


def _make_batch_job(db, workspace_id, agent_id, total=2) -> "BatchJob":  # noqa: F821
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
        voicemail_action="skip",
    )
    db.add(j)
    db.commit()
    db.refresh(j)
    return j


def _make_record(db, batch_job_id, phone="+61412345678", status="waiting"):
    from app.models.batch_call_record import BatchCallRecord

    r = BatchCallRecord(batch_job_id=batch_job_id, phone_number=phone, status=status, attempts=0)
    db.add(r)
    db.commit()
    db.refresh(r)
    return r


# ── BatchCallService.rotate_number_if_flagged ───────────────────────────────

class TestRotateNumberIfFlagged:
    def test_clean_bound_number_does_not_rotate(self, db):
        from app.services.batch_call_service import BatchCallService

        tenant_id = _make_tenant(db)
        agent_id = _make_agent(db, tenant_id)
        bound = _make_phone_number(db, tenant_id, agent_id, "+61412345678")
        _make_reputation(db, bound.id, spam_flagged=False, score=90)
        job = _make_batch_job(db, tenant_id, agent_id)

        svc = BatchCallService(db)
        result = asyncio_run(svc.rotate_number_if_flagged(tenant_id, agent_id, job.id))

        assert result is None
        db.refresh(job)
        assert job.actual_from_number is None

    def test_rotates_to_clean_number_with_same_area_code(self, db):
        from app.services.batch_call_service import BatchCallService

        tenant_id = _make_tenant(db)
        agent_id = _make_agent(db, tenant_id)
        bound = _make_phone_number(db, tenant_id, agent_id, "+61412345678")  # AU, area 412
        _make_reputation(db, bound.id, spam_flagged=True, score=10)

        same_area = _make_extra_number(db, tenant_id, "+61412999888")  # same area 412
        _make_extra_number(db, tenant_id, "+61355566677")  # same country, area 355

        job = _make_batch_job(db, tenant_id, agent_id)

        svc = BatchCallService(db)
        result = asyncio_run(svc.rotate_number_if_flagged(tenant_id, agent_id, job.id))

        assert result == ("+61412345678", "+61412999888")
        db.refresh(job)
        assert job.actual_from_number == "+61412999888"

    def test_falls_back_to_same_country_when_no_area_code_match(self, db):
        from app.services.batch_call_service import BatchCallService

        tenant_id = _make_tenant(db)
        agent_id = _make_agent(db, tenant_id)
        bound = _make_phone_number(db, tenant_id, agent_id, "+61412345678")
        _make_reputation(db, bound.id, spam_flagged=True, score=10)

        _make_extra_number(db, tenant_id, "+61355566677")

        job = _make_batch_job(db, tenant_id, agent_id)

        svc = BatchCallService(db)
        result = asyncio_run(svc.rotate_number_if_flagged(tenant_id, agent_id, job.id))

        assert result == ("+61412345678", "+61355566677")

    def test_never_rotates_onto_a_number_bound_to_another_agent(self, db):
        """A number already assigned to a different agent must never be picked
        as a replacement — doing so would hijack that agent's caller ID and
        misroute its inbound callbacks/voicemail replies."""
        from app.services.batch_call_service import AllNumbersFlaggedError, BatchCallService

        tenant_id = _make_tenant(db)
        agent_id = _make_agent(db, tenant_id)
        # A second agent in the same tenant — only its id is needed to bind
        # the "other" phone number, so avoid the single-agent-per-tenant
        # uniqueness quirk in the SQLite test schema by not persisting it.
        other_agent_id = uuid.uuid4()
        bound = _make_phone_number(db, tenant_id, agent_id, "+61412345678")
        _make_reputation(db, bound.id, spam_flagged=True, score=10)

        # Only candidate in the pool is bound to a different agent.
        _make_phone_number(db, tenant_id, other_agent_id, "+61412999888")

        job = _make_batch_job(db, tenant_id, agent_id)

        svc = BatchCallService(db)
        with pytest.raises(AllNumbersFlaggedError):
            asyncio_run(svc.rotate_number_if_flagged(tenant_id, agent_id, job.id))

    def test_raises_and_fails_job_when_no_clean_number_available(self, db):
        from app.services.batch_call_service import AllNumbersFlaggedError, BatchCallService

        tenant_id = _make_tenant(db)
        agent_id = _make_agent(db, tenant_id)
        bound = _make_phone_number(db, tenant_id, agent_id, "+61412345678")
        _make_reputation(db, bound.id, spam_flagged=True, score=10)

        # Another number exists but is also flagged — not usable.
        other = _make_extra_number(db, tenant_id, "+61412999888")
        _make_reputation(db, other.id, spam_flagged=True, score=5)

        job = _make_batch_job(db, tenant_id, agent_id)
        rec = _make_record(db, job.id)

        svc = BatchCallService(db)
        with pytest.raises(AllNumbersFlaggedError):
            asyncio_run(svc.rotate_number_if_flagged(tenant_id, agent_id, job.id))

        db.refresh(job)
        db.refresh(rec)
        assert job.status == "failed"
        assert job.actual_from_number is None
        assert rec.status == "cancelled"

    def test_checks_reputation_when_no_record_exists_yet(self, db):
        """No PhoneNumberReputation row yet — a check is performed to populate it."""
        from app.services.batch_call_service import BatchCallService

        tenant_id = _make_tenant(db)
        agent_id = _make_agent(db, tenant_id)
        bound = _make_phone_number(db, tenant_id, agent_id, "+61412345678")
        job = _make_batch_job(db, tenant_id, agent_id)

        async def _fake_check(db_, phone_number_obj):
            from app.models.phone_number_reputation import PhoneNumberReputation

            row = PhoneNumberReputation(
                phone_number_id=phone_number_obj.id,
                reputation_score=20,
                spam_flagged=True,
                last_checked_at=datetime.now(timezone.utc),
                checked_by="mock",
            )
            db_.add(row)
            db_.commit()
            return {"spam_flagged": True, "reputation_score": 20, "flagged_reason": "low score"}

        clean = _make_extra_number(db, tenant_id, "+61412999888")

        svc = BatchCallService(db)
        with patch(
            "app.services.reputation_service.check_number_reputation",
            new=AsyncMock(side_effect=_fake_check),
        ):
            result = asyncio_run(svc.rotate_number_if_flagged(tenant_id, agent_id, job.id))

        assert result == ("+61412345678", "+61412999888")


# ── POST /batch-calls rotation integration (router level, mocked service) ──

WORKSPACE_ID = uuid.uuid4()
AGENT_ID = uuid.uuid4()
RAW_KEY = f"tgs-{uuid.uuid4().hex}"
AUTH_HEADERS = {"x-api-key": RAW_KEY, "x-workspace-id": str(WORKSPACE_ID)}


def _mock_workspace():
    from app.core.workspace import Workspace

    ws = MagicMock(spec=Workspace)
    ws.id = WORKSPACE_ID
    ws.status = "active"
    ws.is_active = True
    return ws


def _mock_principal():
    from app.core.request_auth import ApiKeyPrincipal

    return ApiKeyPrincipal(current_tenant_id=WORKSPACE_ID, api_key_id=uuid.uuid4())


def _job_out(total: int = 2):
    from app.schemas.batch_call import BatchJobOut

    return BatchJobOut(
        id=uuid.uuid4(),
        workspace_id=WORKSPACE_ID,
        agent_id=AGENT_ID,
        status="pending",
        total_count=total,
        waiting_count=total,
        active_count=0,
        completed_count=0,
        failed_count=0,
        voicemail_action="skip",
        voicemail_message=None,
        s3_path="batch-files/x/y.csv",
        scheduled_at=None,
        started_at=None,
        completed_at=None,
        created_at=datetime.now(timezone.utc),
    )


def _build_app(svc_mock) -> TestClient:
    from app.api.deps import get_workspace, require_tenant
    from app.api.v2.routers import batch_calls as bc_module
    from app.core.exception_handlers import register_exception_handlers

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(bc_module.router)

    mini.dependency_overrides[require_tenant] = _mock_principal
    mini.dependency_overrides[get_workspace] = _mock_workspace

    def _svc_override():
        yield svc_mock

    mini.dependency_overrides[bc_module._batch_service] = _svc_override
    mini.dependency_overrides[bc_module._batch_service_write] = _svc_override

    return TestClient(mini, raise_server_exceptions=False)


_VALID_CSV = b"phone_number\n+15551111001\n+15551111002"


class TestBatchCallsRotationEndpoint:
    @patch("app.api.v2.routers.batch_calls._enqueue_batch_job", new_callable=AsyncMock)
    def test_all_numbers_flagged_returns_422(self, mock_enqueue):
        from app.services.batch_call_service import AllNumbersFlaggedError

        svc = MagicMock()
        svc.create_batch_job.return_value = _job_out()
        svc.rotate_number_if_flagged = AsyncMock(side_effect=AllNumbersFlaggedError())

        client = _build_app(svc)
        resp = client.post(
            "/batch-calls",
            data={"agent_id": str(AGENT_ID)},
            files={"file": ("t.csv", io.BytesIO(_VALID_CSV), "text/csv")},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 422
        body = resp.json()
        assert body["error"]["code"] == "all_numbers_flagged"
        assert "spam-flagged" in body["error"]["message"]
        mock_enqueue.assert_not_called()

    @patch("app.api.v2.routers.batch_calls._enqueue_batch_job", new_callable=AsyncMock)
    @patch("app.api.v2.routers.batch_calls.log_audit_event")
    def test_rotation_logs_audit_event_and_still_enqueues(self, mock_audit, mock_enqueue):
        job = _job_out()
        svc = MagicMock()
        svc.create_batch_job.return_value = job
        svc.rotate_number_if_flagged = AsyncMock(return_value=("+15550001111", "+15550002222"))

        client = _build_app(svc)
        resp = client.post(
            "/batch-calls",
            data={"agent_id": str(AGENT_ID)},
            files={"file": ("t.csv", io.BytesIO(_VALID_CSV), "text/csv")},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 201
        mock_enqueue.assert_called_once()

        rotation_calls = [
            c for c in mock_audit.call_args_list if c.kwargs.get("action") == "batch.number_rotated"
        ]
        assert len(rotation_calls) == 1
        call = rotation_calls[0]
        assert call.kwargs["old_value"] == {"from_number": "+15550001111"}
        assert call.kwargs["new_value"] == {"from_number": "+15550002222", "reason": "spam_flagged"}

    @patch("app.api.v2.routers.batch_calls._enqueue_batch_job", new_callable=AsyncMock)
    @patch("app.api.v2.routers.batch_calls.log_audit_event")
    def test_no_rotation_when_bound_number_clean(self, mock_audit, mock_enqueue):
        job = _job_out()
        svc = MagicMock()
        svc.create_batch_job.return_value = job
        svc.rotate_number_if_flagged = AsyncMock(return_value=None)

        client = _build_app(svc)
        resp = client.post(
            "/batch-calls",
            data={"agent_id": str(AGENT_ID)},
            files={"file": ("t.csv", io.BytesIO(_VALID_CSV), "text/csv")},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 201
        rotation_calls = [
            c for c in mock_audit.call_args_list if c.kwargs.get("action") == "batch.number_rotated"
        ]
        assert len(rotation_calls) == 0


# ── ARQ cron: check_all_phone_numbers_reputation ────────────────────────────

class TestReputationCronTask:
    @patch("app.services.reputation_service.check_number_reputation", new_callable=AsyncMock)
    def test_checks_only_missing_or_stale_numbers(self, mock_check, db, monkeypatch):
        from app.workers.batch_call_worker import check_all_phone_numbers_reputation

        tenant_id = _make_tenant(db)
        agent_id = _make_agent(db, tenant_id)

        never_checked = _make_phone_number(db, tenant_id, agent_id, "+61412000001")

        stale = _make_extra_number(db, tenant_id, "+61412000002")
        _make_reputation(
            db, stale.id, spam_flagged=False, score=95,
            checked_at=datetime.now(timezone.utc) - timedelta(hours=25),
        )

        fresh = _make_extra_number(db, tenant_id, "+61412000003")
        _make_reputation(
            db, fresh.id, spam_flagged=False, score=95,
            checked_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )

        inactive = _make_extra_number(db, tenant_id, "+61412000004")
        inactive.status = "inactive"
        db.commit()

        monkeypatch.setattr("app.db.session.SessionLocal", lambda: db)
        # db.close() is called by the task; make it a no-op so the fixture's
        # session stays open for post-assertions.
        monkeypatch.setattr(db, "close", lambda: None)

        asyncio_run(check_all_phone_numbers_reputation({}))

        checked_ids = {call.args[1].id for call in mock_check.call_args_list}
        assert never_checked.id in checked_ids
        assert stale.id in checked_ids
        assert fresh.id not in checked_ids
        assert inactive.id not in checked_ids
        assert mock_check.call_count == 2


# ── asyncio helper ────────────────────────────────────────────────────────────

def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)
