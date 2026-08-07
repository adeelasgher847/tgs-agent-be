"""Unit tests for the automatic call-summary-generation hook wired into
CallSessionService.update_call_session_status().

Unlike the HubSpot/Salesforce/GHL post-call write-back hooks, this hook must
fire for demo-link (web) calls too, since a demo call still has a real
transcript worth summarizing even though it has no CRM contact to write back
to. See app/services/voice_analysis_service.py::schedule_call_summary_generation.

Fixture setup mirrors tests/services/test_call_session_demo_link_usage.py.
"""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from app.models.agent import Agent
from app.models.call_session import CallSession
from app.models.tenant import Tenant
from app.models.user import User
from app.services.call_session_service import call_session_service


@contextmanager
def _naive_utcnow_patch():
    """See tests/services/test_call_session_demo_link_usage.py for the full
    rationale: SQLite doesn't round-trip tzinfo through DateTime(timezone=True),
    so a CallSession reloaded from the test DB always has a naive start_time.
    Patch datetime.now(tz) inside call_session_service to also return a naive
    value so the real duration-calculation code path can still be exercised.
    """
    real_datetime = datetime

    class _NaiveNow(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime.utcnow()

    with patch("app.services.call_session_service.datetime", _NaiveNow):
        yield


@pytest.fixture
def tenant(db) -> Tenant:
    t = Tenant(
        name=f"SummaryHookWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"summary_hook_ws_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@pytest.fixture
def agent(db, tenant: Tenant) -> Agent:
    a = Agent(
        tenant_id=tenant.id,
        name="Summary Hook Agent",
        status="active",
        llm_model="gpt-4o-mini",
        tts_provider_slug="elevenlabs",
        tts_voice_external_id="voice-x",
        tts_language="en",
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    return a


@pytest.fixture
def cs_user(db, tenant: Tenant) -> User:
    u = User(
        email=f"summary-hook-user-{uuid.uuid4().hex[:8]}@example.com",
        first_name="Summary",
        last_name="Hook",
        hashed_password="",
        current_tenant_id=tenant.id,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _make_session(
    db, tenant: Tenant, agent: Agent, user: User, *, call_metadata: dict | None, minutes_ago: int = 5
) -> CallSession:
    session = CallSession(
        user_id=user.id,
        agent_id=agent.id,
        tenant_id=tenant.id,
        status="active",
        call_type="web",
        start_time=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
        call_metadata=call_metadata,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@pytest.mark.usefixtures("db")
class TestAutomaticCallSummaryGenerationHook:
    def test_completed_call_schedules_summary_generation(self, db, tenant, agent, cs_user):
        session = _make_session(db, tenant, agent, cs_user, call_metadata=None)

        with _naive_utcnow_patch(), patch(
            "app.services.voice_analysis_service.schedule_call_summary_generation"
        ) as mock_schedule:
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        assert updated.status == "completed"
        mock_schedule.assert_called_once_with(session.id)

    def test_completed_demo_link_call_still_schedules_summary_generation(
        self, db, tenant, agent, cs_user
    ):
        """Unlike the CRM write-back hooks, this must NOT be skipped for
        demo-link (web) calls."""
        session = _make_session(
            db, tenant, agent, cs_user,
            call_metadata={
                "demo_link": {
                    "demo_link_id": str(uuid.uuid4()),
                    "visitor_id": "visitor-summary-hook",
                }
            },
        )

        with _naive_utcnow_patch(), patch(
            "app.services.voice_analysis_service.schedule_call_summary_generation"
        ) as mock_schedule:
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        assert updated.status == "completed"
        mock_schedule.assert_called_once_with(session.id)

    def test_schedule_failure_is_swallowed_and_status_update_still_succeeds(
        self, db, tenant, agent, cs_user
    ):
        """Fail-open: an exception raised inside schedule_call_summary_generation
        must not roll back the status update or propagate to the caller."""
        session = _make_session(db, tenant, agent, cs_user, call_metadata=None)

        with _naive_utcnow_patch(), patch(
            "app.services.voice_analysis_service.schedule_call_summary_generation",
            side_effect=RuntimeError("simulated ARQ enqueue failure"),
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        assert updated.status == "completed"
        assert updated.duration is not None

    @pytest.mark.parametrize("status", ["failed", "no_answer", "ringing"])
    def test_non_completed_status_does_not_schedule_summary_generation(
        self, db, tenant, agent, cs_user, status
    ):
        session = _make_session(db, tenant, agent, cs_user, call_metadata=None)

        with _naive_utcnow_patch(), patch(
            "app.services.voice_analysis_service.schedule_call_summary_generation"
        ) as mock_schedule:
            updated = call_session_service.update_call_session_status(
                db, session.id, status
            )

        assert updated is not None
        assert updated.status == status
        mock_schedule.assert_not_called()

    def test_completed_inbound_call_with_trello_sync_skips_duplicate_summary_job(
        self, db, tenant, agent, cs_user
    ):
        """When inbound CRM sync (Trello) is already going to run
        analyze_call_transcript for this session, the summary hook must not
        also enqueue a second, independent ARQ job — that would race two
        jobs into duplicate (paid) LLM calls instead of one cached analysis.
        """
        session = _make_session(db, tenant, agent, cs_user, call_metadata=None)
        session.call_type = "inbound"
        db.commit()

        with (
            _naive_utcnow_patch(),
            patch(
                "app.services.call_session_service.tenant_has_active_inbound_crm",
                return_value=True,
            ),
            patch(
                "app.services.call_session_service.schedule_inbound_crm_sync"
            ) as mock_inbound_sync,
            patch(
                "app.services.voice_analysis_service.schedule_call_summary_generation"
            ) as mock_schedule,
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        assert updated.status == "completed"
        mock_inbound_sync.assert_called_once_with(session.id)
        mock_schedule.assert_not_called()

    def test_completed_inbound_call_without_trello_sync_still_schedules_summary(
        self, db, tenant, agent, cs_user
    ):
        """Inbound calls for tenants without active inbound-CRM sync must
        still get the automatic summary job — the skip is specific to the
        case where inbound CRM sync will run the analysis itself."""
        session = _make_session(db, tenant, agent, cs_user, call_metadata=None)
        session.call_type = "inbound"
        db.commit()

        with (
            _naive_utcnow_patch(),
            patch(
                "app.services.call_session_service.tenant_has_active_inbound_crm",
                return_value=False,
            ),
            patch(
                "app.services.voice_analysis_service.schedule_call_summary_generation"
            ) as mock_schedule,
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        assert updated.status == "completed"
        mock_schedule.assert_called_once_with(session.id)
