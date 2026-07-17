"""
Unit tests for the Twilio AMD webhook (app/routers/amd_webhook.py).

DB is an in-memory SQLite fixture (same pattern as tests/services/test_batch_call_worker.py).
Twilio REST calls are mocked at the twilio_service boundary — no network calls.
Signature validation is bypassed via ALLOW_UNAUTHENTICATED_WEBHOOKS except where
specifically under test.
"""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.base import Base


@compiles(JSONB, "sqlite")
def _jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


@compiles(PG_UUID, "sqlite")
def _uuid_sqlite(type_, compiler, **kw):
    return "VARCHAR(36)"


_SHARED_CONN = sqlite3.connect(":memory:", check_same_thread=False)
_engine = create_engine(
    "sqlite://", creator=lambda: _SHARED_CONN, connect_args={"check_same_thread": False}
)
_Session = sessionmaker(bind=_engine)


def _setup_db():
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


@pytest.fixture(autouse=True)
def _bypass_signature_validation():
    """Most tests exercise business logic, not signature validation — bypass it
    by default; TestSignatureValidation overrides this explicitly."""
    with patch.object(settings, "ALLOW_UNAUTHENTICATED_WEBHOOKS", True):
        yield


class _FakeRequest:
    def __init__(self, form_data: dict):
        self._form_data = form_data
        self.headers = {}

    async def form(self):
        return self._form_data


def _make_tenant(db):
    from app.models.tenant import Tenant

    t = Tenant(name="WS", schema_name="ws_test")
    db.add(t)
    db.commit()
    db.refresh(t)
    return t.id


def _make_agent(db, tenant_id):
    from app.models.agent import Agent

    a = Agent(tenant_id=tenant_id, name="TestAgent", system_prompt="", status="ready")
    db.add(a)
    db.commit()
    db.refresh(a)
    return a.id


def _make_user(db):
    from app.models.user import User

    u = User(
        id=uuid.uuid4(),
        first_name="AMD",
        last_name="Test",
        email=f"amd-{uuid.uuid4().hex}@example.com",
        hashed_password="x",
    )
    db.add(u)
    db.flush()
    return u.id


def _make_call_session(db, tenant_id, agent_id, user_id, start_time=None):
    from app.models.call_session import CallSession

    cs = CallSession(
        id=uuid.uuid4(),
        user_id=user_id,
        agent_id=agent_id,
        tenant_id=tenant_id,
        start_time=start_time or datetime.now(timezone.utc),
        status="active",
        call_type="outbound",
        twilio_call_sid="CAtest123",
    )
    db.add(cs)
    db.commit()
    db.refresh(cs)
    return cs


def _make_batch_job(
    db, workspace_id, agent_id, voicemail_action="skip", voicemail_message=None
):
    from app.models.batch_job import BatchJob

    j = BatchJob(
        workspace_id=workspace_id,
        agent_id=agent_id,
        status="processing",
        total_count=1,
        waiting_count=0,
        active_count=1,
        completed_count=0,
        failed_count=0,
        voicemail_action=voicemail_action,
        voicemail_message=voicemail_message,
    )
    db.add(j)
    db.commit()
    db.refresh(j)
    return j


def _make_record(db, batch_job_id, call_id):
    from app.models.batch_call_record import BatchCallRecord

    r = BatchCallRecord(
        batch_job_id=batch_job_id,
        phone_number="+15550001111",
        status="active",
        call_id=call_id,
        attempts=1,
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    return r


# ── machine_start ──────────────────────────────────────────────────────────────


class TestMachineStart:
    def test_skip_action_hangs_up_and_marks_skipped(self, db):
        from app.routers.amd_webhook import amd_callback

        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        user_id = _make_user(db)
        cs = _make_call_session(db, workspace_id, agent_id, user_id)
        job = _make_batch_job(db, workspace_id, agent_id, voicemail_action="skip")
        record = _make_record(db, job.id, cs.id)

        request = _FakeRequest({"AnsweredBy": "machine_start", "CallSid": "CAtest123"})

        with patch("app.routers.amd_webhook.twilio_service.end_call") as mock_end:
            asyncio.run(
                amd_callback(
                    request=request,
                    callSessionId=str(cs.id),
                    batchCallRecordId=str(record.id),
                    db=db,
                )
            )
            mock_end.assert_called_once_with("CAtest123")

        db.refresh(record)
        db.refresh(job)
        assert record.status == "voicemail_skipped"
        assert job.voicemail_skipped_count == 1
        assert job.active_count == 0

    def test_leave_message_action_does_not_hang_up_yet(self, db):
        from app.routers.amd_webhook import amd_callback

        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        user_id = _make_user(db)
        cs = _make_call_session(db, workspace_id, agent_id, user_id)
        job = _make_batch_job(
            db,
            workspace_id,
            agent_id,
            voicemail_action="leave_message",
            voicemail_message="We tried to reach you.",
        )
        record = _make_record(db, job.id, cs.id)

        request = _FakeRequest({"AnsweredBy": "machine_start", "CallSid": "CAtest123"})

        with patch(
            "app.routers.amd_webhook.twilio_service.end_call"
        ) as mock_end, patch(
            "app.routers.amd_webhook.twilio_service.update_call_twiml"
        ) as mock_update:
            asyncio.run(
                amd_callback(
                    request=request,
                    callSessionId=str(cs.id),
                    batchCallRecordId=str(record.id),
                    db=db,
                )
            )
            mock_end.assert_not_called()
            mock_update.assert_not_called()

        db.refresh(record)
        db.refresh(cs)
        assert record.status == "active"  # unchanged — waiting for the beep
        assert cs.call_metadata["amd_result"] == "machine_start"

    def test_continue_action_does_not_hang_up_and_releases_hold(self, db):
        """voicemail_action='continue' must let the call proceed into the normal
        flow instead of hanging up or getting stuck (regression guard)."""
        from app.routers.amd_webhook import amd_callback

        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        user_id = _make_user(db)
        cs = _make_call_session(db, workspace_id, agent_id, user_id)
        job = _make_batch_job(db, workspace_id, agent_id, voicemail_action="continue")
        record = _make_record(db, job.id, cs.id)

        request = _FakeRequest({"AnsweredBy": "machine_start", "CallSid": "CAtest123"})

        with patch("app.routers.amd_webhook.twilio_service.end_call") as mock_end:
            asyncio.run(
                amd_callback(
                    request=request,
                    callSessionId=str(cs.id),
                    batchCallRecordId=str(record.id),
                    db=db,
                )
            )
            mock_end.assert_not_called()

        db.refresh(cs)
        db.refresh(record)
        assert cs.call_metadata["amd_result"] == "continue"
        assert record.status == "active"

    def test_missing_job_defaults_to_continue_not_skip(self, db):
        """A non-batch call that somehow set enable_amd (no batchCallRecordId
        resolves) must fail open instead of silently hanging up a real caller."""
        from app.routers.amd_webhook import amd_callback

        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        user_id = _make_user(db)
        cs = _make_call_session(db, workspace_id, agent_id, user_id)

        request = _FakeRequest({"AnsweredBy": "machine_start", "CallSid": "CAtest123"})

        with patch("app.routers.amd_webhook.twilio_service.end_call") as mock_end:
            asyncio.run(
                amd_callback(
                    request=request,
                    callSessionId=str(cs.id),
                    batchCallRecordId=None,
                    db=db,
                )
            )
            mock_end.assert_not_called()

        db.refresh(cs)
        assert cs.call_metadata["amd_result"] == "continue"


# ── machine_end_beep ─────────────────────────────────────────────────────────


class TestMachineEndBeep:
    def test_leave_message_plays_tts_and_marks_message_left(self, db):
        from app.routers.amd_webhook import amd_callback

        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        user_id = _make_user(db)
        cs = _make_call_session(db, workspace_id, agent_id, user_id)
        job = _make_batch_job(
            db,
            workspace_id,
            agent_id,
            voicemail_action="leave_message",
            voicemail_message="We tried to reach you.",
        )
        record = _make_record(db, job.id, cs.id)

        request = _FakeRequest(
            {"AnsweredBy": "machine_end_beep", "CallSid": "CAtest123"}
        )

        with patch(
            "app.routers.amd_webhook.twilio_service.update_call_twiml"
        ) as mock_update, patch(
            "app.services.billing_service.BillingService.record_call_usage"
        ):
            asyncio.run(
                amd_callback(
                    request=request,
                    callSessionId=str(cs.id),
                    batchCallRecordId=str(record.id),
                    db=db,
                )
            )
            mock_update.assert_called_once()
            call_sid_arg, twiml_arg = (
                mock_update.call_args.args[0],
                mock_update.call_args.args[1],
            )
            assert call_sid_arg == "CAtest123"
            assert "Play" in twiml_arg
            assert "Hangup" in twiml_arg

        db.refresh(record)
        db.refresh(job)
        assert record.status == "voicemail_message_left"
        assert job.voicemail_message_left_count == 1

    def test_skip_action_ignores_beep(self, db):
        """If voicemail_action is 'skip' the call was already hung up on machine_start;
        a stray beep callback must not attempt to play anything."""
        from app.routers.amd_webhook import amd_callback

        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        user_id = _make_user(db)
        cs = _make_call_session(db, workspace_id, agent_id, user_id)
        job = _make_batch_job(db, workspace_id, agent_id, voicemail_action="skip")
        record = _make_record(db, job.id, cs.id)

        request = _FakeRequest(
            {"AnsweredBy": "machine_end_beep", "CallSid": "CAtest123"}
        )

        with patch(
            "app.routers.amd_webhook.twilio_service.update_call_twiml"
        ) as mock_update:
            asyncio.run(
                amd_callback(
                    request=request,
                    callSessionId=str(cs.id),
                    batchCallRecordId=str(record.id),
                    db=db,
                )
            )
            mock_update.assert_not_called()

    def test_leave_message_with_no_configured_message_falls_back_to_skip(self, db):
        """An empty voicemail_message means there's nothing to play — must hang
        up and count as skipped, not falsely report a message was left."""
        from app.routers.amd_webhook import amd_callback

        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        user_id = _make_user(db)
        cs = _make_call_session(db, workspace_id, agent_id, user_id)
        job = _make_batch_job(
            db,
            workspace_id,
            agent_id,
            voicemail_action="leave_message",
            voicemail_message=None,
        )
        record = _make_record(db, job.id, cs.id)

        request = _FakeRequest(
            {"AnsweredBy": "machine_end_beep", "CallSid": "CAtest123"}
        )

        with patch(
            "app.routers.amd_webhook.twilio_service.update_call_twiml"
        ) as mock_update, patch(
            "app.routers.amd_webhook.twilio_service.end_call"
        ) as mock_end:
            asyncio.run(
                amd_callback(
                    request=request,
                    callSessionId=str(cs.id),
                    batchCallRecordId=str(record.id),
                    db=db,
                )
            )
            mock_update.assert_not_called()
            mock_end.assert_called_once_with("CAtest123")

        db.refresh(record)
        db.refresh(job)
        assert record.status == "voicemail_skipped"
        assert job.voicemail_skipped_count == 1
        assert job.voicemail_message_left_count == 0


# ── human / unknown ────────────────────────────────────────────────────────────


class TestHumanAnswered:
    def test_human_persists_amd_result_and_does_not_touch_record(self, db):
        from app.routers.amd_webhook import amd_callback

        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        user_id = _make_user(db)
        cs = _make_call_session(db, workspace_id, agent_id, user_id)
        job = _make_batch_job(db, workspace_id, agent_id, voicemail_action="skip")
        record = _make_record(db, job.id, cs.id)

        request = _FakeRequest({"AnsweredBy": "human", "CallSid": "CAtest123"})

        with patch("app.routers.amd_webhook.twilio_service.end_call") as mock_end:
            asyncio.run(
                amd_callback(
                    request=request,
                    callSessionId=str(cs.id),
                    batchCallRecordId=str(record.id),
                    db=db,
                )
            )
            mock_end.assert_not_called()

        db.refresh(cs)
        db.refresh(record)
        assert cs.call_metadata["amd_result"] == "human"
        assert record.status == "active"  # normal call flow continues untouched


# ── signature validation ──────────────────────────────────────────────────────


class TestSignatureValidation:
    def test_amd_callback_rejects_unsigned_request_when_not_bypassed(self, db):
        from fastapi import HTTPException

        from app.routers.amd_webhook import amd_callback

        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        user_id = _make_user(db)
        cs = _make_call_session(db, workspace_id, agent_id, user_id)

        request = _FakeRequest({"AnsweredBy": "human", "CallSid": "CAtest123"})

        with patch.object(settings, "ALLOW_UNAUTHENTICATED_WEBHOOKS", False), patch(
            "app.routers.amd_webhook.validate_twilio_signature", return_value=False
        ), patch(
            "app.routers.amd_webhook.validate_twilio_signature_with_token",
            return_value=False,
        ):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(
                    amd_callback(
                        request=request,
                        callSessionId=str(cs.id),
                        batchCallRecordId=None,
                        db=db,
                    )
                )

        assert exc_info.value.status_code == 403

    def test_amd_hold_rejects_unsigned_request_when_not_bypassed(self, db):
        from fastapi import HTTPException

        from app.routers.amd_webhook import amd_hold

        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        user_id = _make_user(db)
        cs = _make_call_session(db, workspace_id, agent_id, user_id)

        with patch.object(settings, "ALLOW_UNAUTHENTICATED_WEBHOOKS", False), patch(
            "app.routers.amd_webhook.validate_twilio_signature", return_value=False
        ), patch(
            "app.routers.amd_webhook.validate_twilio_signature_with_token",
            return_value=False,
        ):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(
                    amd_hold(
                        request=_FakeRequest({}),
                        agentId=str(agent_id),
                        userId=str(user_id),
                        callSessionId=str(cs.id),
                        db=db,
                    )
                )

        assert exc_info.value.status_code == 403


# ── amd-hold ──────────────────────────────────────────────────────────────────


class TestAmdHold:
    def test_redirects_to_streaming_once_human_result_recorded(self, db):
        from app.routers.amd_webhook import amd_hold

        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        user_id = _make_user(db)
        cs = _make_call_session(db, workspace_id, agent_id, user_id)
        cs.call_metadata = {"amd_result": "human"}
        db.commit()

        resp = asyncio.run(
            amd_hold(
                request=_FakeRequest({}),
                agentId=str(agent_id),
                userId=str(user_id),
                callSessionId=str(cs.id),
                db=db,
            )
        )

        assert "gather/streaming" in resp.body.decode()
        assert "Redirect" in resp.body.decode()

    def test_redirects_to_streaming_when_continue_result_recorded(self, db):
        from app.routers.amd_webhook import amd_hold

        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        user_id = _make_user(db)
        cs = _make_call_session(db, workspace_id, agent_id, user_id)
        cs.call_metadata = {"amd_result": "continue"}
        db.commit()

        resp = asyncio.run(
            amd_hold(
                request=_FakeRequest({}),
                agentId=str(agent_id),
                userId=str(user_id),
                callSessionId=str(cs.id),
                db=db,
            )
        )

        assert "gather/streaming" in resp.body.decode()

    def test_pauses_and_loops_when_amd_result_not_yet_known(self, db):
        from app.routers.amd_webhook import amd_hold

        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        user_id = _make_user(db)
        cs = _make_call_session(db, workspace_id, agent_id, user_id)

        resp = asyncio.run(
            amd_hold(
                request=_FakeRequest({}),
                agentId=str(agent_id),
                userId=str(user_id),
                callSessionId=str(cs.id),
                db=db,
            )
        )

        body = resp.body.decode()
        assert "Pause" in body
        assert "amd-hold" in body

    def test_releases_hold_after_max_wait_even_without_amd_result(self, db):
        """Guards against an indefinite hold if the async AMD callback never arrives."""
        from app.routers.amd_webhook import amd_hold

        workspace_id = _make_tenant(db)
        agent_id = _make_agent(db, workspace_id)
        user_id = _make_user(db)
        stale_start = datetime.now(timezone.utc) - timedelta(seconds=60)
        cs = _make_call_session(
            db, workspace_id, agent_id, user_id, start_time=stale_start
        )

        resp = asyncio.run(
            amd_hold(
                request=_FakeRequest({}),
                agentId=str(agent_id),
                userId=str(user_id),
                callSessionId=str(cs.id),
                db=db,
            )
        )

        body = resp.body.decode()
        assert "gather/streaming" in body
        assert "Pause" not in body

    def test_fails_open_to_streaming_when_call_session_not_found(self, db):
        from app.routers.amd_webhook import amd_hold

        resp = asyncio.run(
            amd_hold(
                request=_FakeRequest({}),
                agentId="agent-x",
                userId="user-x",
                callSessionId=str(uuid.uuid4()),
                db=db,
            )
        )

        assert "gather/streaming" in resp.body.decode()
