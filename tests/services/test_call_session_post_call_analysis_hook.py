"""Unit tests for the post-call analysis hook wired into
CallSessionService.update_call_session_status().

Mirrors tests/services/test_call_session_post_call_email_hook.py's fixture
setup and mocking conventions. This hook fires purely off
call_flow.post_call_analysis_variables being non-empty — there's no explicit
enable toggle.
"""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from app.models.agent import Agent
from app.models.call_flow import CallFlow
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
        name=f"AnalysisHookWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"analysis_hook_ws_{uuid.uuid4().hex[:8]}",
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
        name="Analysis Hook Agent",
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
        email=f"analysis-hook-user-{uuid.uuid4().hex[:8]}@example.com",
        first_name="Analysis",
        last_name="Hook",
        hashed_password="",
        current_tenant_id=tenant.id,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _make_flow(
    db,
    tenant: Tenant,
    agent: Agent,
    *,
    post_call_analysis_variables: list | None = None,
) -> CallFlow:
    f = CallFlow(
        tenant_id=tenant.id,
        agent_id=agent.id,
        name="Analysis Hook Flow",
        direction="inbound",
        post_call_analysis_variables=post_call_analysis_variables or [],
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    return f


def _make_session(
    db,
    tenant: Tenant,
    agent: Agent,
    user: User,
    *,
    call_flow: CallFlow | None = None,
    minutes_ago: int = 5,
) -> CallSession:
    session = CallSession(
        user_id=user.id,
        agent_id=agent.id,
        tenant_id=tenant.id,
        call_flow_id=call_flow.id if call_flow else None,
        status="active",
        call_type="web",
        start_time=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@pytest.mark.usefixtures("db")
class TestPostCallAnalysisHook:
    def test_completed_call_with_variables_configured_schedules_job(
        self, db, tenant, agent, cs_user
    ):
        flow = _make_flow(
            db,
            tenant,
            agent,
            post_call_analysis_variables=[
                {"name": "service_type", "description": "Extract the service type."}
            ],
        )
        session = _make_session(db, tenant, agent, cs_user, call_flow=flow)

        with (
            _naive_utcnow_patch(),
            patch("app.services.voice_analysis_service.schedule_call_summary_generation"),
            patch("app.services.post_call_email_service.schedule_post_call_email_summary"),
            patch(
                "app.services.post_call_analysis_service.schedule_run_post_call_analysis"
            ) as mock_schedule,
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        assert updated.status == "completed"
        mock_schedule.assert_called_once_with(session.id)

    def test_completed_call_with_no_variables_does_not_schedule(
        self, db, tenant, agent, cs_user
    ):
        flow = _make_flow(db, tenant, agent, post_call_analysis_variables=[])
        session = _make_session(db, tenant, agent, cs_user, call_flow=flow)

        with (
            _naive_utcnow_patch(),
            patch("app.services.voice_analysis_service.schedule_call_summary_generation"),
            patch("app.services.post_call_email_service.schedule_post_call_email_summary"),
            patch(
                "app.services.post_call_analysis_service.schedule_run_post_call_analysis"
            ) as mock_schedule,
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        assert updated.status == "completed"
        mock_schedule.assert_not_called()

    def test_completed_call_with_no_call_flow_does_not_schedule(
        self, db, tenant, agent, cs_user
    ):
        session = _make_session(db, tenant, agent, cs_user, call_flow=None)

        with (
            _naive_utcnow_patch(),
            patch("app.services.voice_analysis_service.schedule_call_summary_generation"),
            patch("app.services.post_call_email_service.schedule_post_call_email_summary"),
            patch(
                "app.services.post_call_analysis_service.schedule_run_post_call_analysis"
            ) as mock_schedule,
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        assert updated.status == "completed"
        mock_schedule.assert_not_called()

    @pytest.mark.parametrize("status", ["failed", "no_answer", "ringing"])
    def test_non_completed_status_does_not_schedule(
        self, db, tenant, agent, cs_user, status
    ):
        flow = _make_flow(
            db,
            tenant,
            agent,
            post_call_analysis_variables=[
                {"name": "service_type", "description": "Extract the service type."}
            ],
        )
        session = _make_session(db, tenant, agent, cs_user, call_flow=flow)

        with (
            _naive_utcnow_patch(),
            patch("app.services.voice_analysis_service.schedule_call_summary_generation"),
            patch("app.services.post_call_email_service.schedule_post_call_email_summary"),
            patch(
                "app.services.post_call_analysis_service.schedule_run_post_call_analysis"
            ) as mock_schedule,
        ):
            updated = call_session_service.update_call_session_status(db, session.id, status)

        assert updated is not None
        assert updated.status == status
        mock_schedule.assert_not_called()

    def test_inbound_crm_sync_call_does_not_schedule(self, db, tenant, agent, cs_user):
        """Same skip condition as its sibling hooks: when this session is an
        inbound-CRM-sync call, the post-call-analysis hook must not fire either."""
        flow = _make_flow(
            db,
            tenant,
            agent,
            post_call_analysis_variables=[
                {"name": "service_type", "description": "Extract the service type."}
            ],
        )
        session = _make_session(db, tenant, agent, cs_user, call_flow=flow)
        session.call_type = "inbound"
        db.commit()

        with (
            _naive_utcnow_patch(),
            patch(
                "app.services.call_session_service.tenant_has_active_inbound_crm",
                return_value=True,
            ),
            patch("app.services.call_session_service.schedule_inbound_crm_sync"),
            patch("app.services.voice_analysis_service.schedule_call_summary_generation"),
            patch("app.services.post_call_email_service.schedule_post_call_email_summary"),
            patch(
                "app.services.post_call_analysis_service.schedule_run_post_call_analysis"
            ) as mock_schedule,
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        assert updated.status == "completed"
        mock_schedule.assert_not_called()

    def test_schedule_failure_is_swallowed_and_status_update_still_succeeds(
        self, db, tenant, agent, cs_user
    ):
        """Fail-open: an exception raised inside schedule_run_post_call_analysis
        must not roll back the status update or propagate to the caller."""
        flow = _make_flow(
            db,
            tenant,
            agent,
            post_call_analysis_variables=[
                {"name": "service_type", "description": "Extract the service type."}
            ],
        )
        session = _make_session(db, tenant, agent, cs_user, call_flow=flow)

        with (
            _naive_utcnow_patch(),
            patch("app.services.voice_analysis_service.schedule_call_summary_generation"),
            patch("app.services.post_call_email_service.schedule_post_call_email_summary"),
            patch(
                "app.services.post_call_analysis_service.schedule_run_post_call_analysis",
                side_effect=RuntimeError("simulated ARQ enqueue failure"),
            ),
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        assert updated.status == "completed"
        assert updated.duration is not None
