"""Unit tests for the post-call email summary hook wired into
CallSessionService.update_call_session_status().

Mirrors tests/services/test_call_session_summary_generation_hook.py's
fixture setup and mocking conventions. Unlike that hook, this one requires
at least one of the two call-flow toggles to be enabled before it fires —
see app/services/post_call_email_service.py::schedule_post_call_email_summary.
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
        name=f"EmailHookWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"email_hook_ws_{uuid.uuid4().hex[:8]}",
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
        name="Email Hook Agent",
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
        email=f"email-hook-user-{uuid.uuid4().hex[:8]}@example.com",
        first_name="Email",
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
    email_summary_enabled: bool = False,
    summary_to_business_owner_enabled: bool = False,
    slack_summary_enabled: bool = False,
) -> CallFlow:
    f = CallFlow(
        tenant_id=tenant.id,
        agent_id=agent.id,
        name="Email Hook Flow",
        direction="inbound",
        email_summary_enabled=email_summary_enabled,
        email_summary_recipients=["x@example.com"] if email_summary_enabled else [],
        summary_to_business_owner_enabled=summary_to_business_owner_enabled,
        slack_summary_enabled=slack_summary_enabled,
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
    call_metadata: dict | None = None,
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
        call_metadata=call_metadata,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@pytest.mark.usefixtures("db")
class TestPostCallEmailSummaryHook:
    def test_completed_call_with_email_summary_enabled_schedules_job(
        self, db, tenant, agent, cs_user
    ):
        flow = _make_flow(db, tenant, agent, email_summary_enabled=True)
        session = _make_session(db, tenant, agent, cs_user, call_flow=flow)

        with (
            _naive_utcnow_patch(),
            patch(
                "app.services.voice_analysis_service.schedule_call_summary_generation"
            ),
            patch(
                "app.services.post_call_email_service.schedule_post_call_email_summary"
            ) as mock_schedule,
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        assert updated.status == "completed"
        mock_schedule.assert_called_once_with(session.id)

    def test_completed_call_with_business_owner_toggle_schedules_job(
        self, db, tenant, agent, cs_user
    ):
        flow = _make_flow(db, tenant, agent, summary_to_business_owner_enabled=True)
        session = _make_session(db, tenant, agent, cs_user, call_flow=flow)

        with (
            _naive_utcnow_patch(),
            patch(
                "app.services.voice_analysis_service.schedule_call_summary_generation"
            ),
            patch(
                "app.services.post_call_email_service.schedule_post_call_email_summary"
            ) as mock_schedule,
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        assert updated.status == "completed"
        mock_schedule.assert_called_once_with(session.id)

    def test_completed_call_with_both_toggles_off_does_not_schedule(
        self, db, tenant, agent, cs_user
    ):
        flow = _make_flow(db, tenant, agent)
        session = _make_session(db, tenant, agent, cs_user, call_flow=flow)

        with (
            _naive_utcnow_patch(),
            patch(
                "app.services.voice_analysis_service.schedule_call_summary_generation"
            ),
            patch(
                "app.services.post_call_email_service.schedule_post_call_email_summary"
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
            patch(
                "app.services.voice_analysis_service.schedule_call_summary_generation"
            ),
            patch(
                "app.services.post_call_email_service.schedule_post_call_email_summary"
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
        flow = _make_flow(db, tenant, agent, email_summary_enabled=True)
        session = _make_session(db, tenant, agent, cs_user, call_flow=flow)

        with (
            _naive_utcnow_patch(),
            patch(
                "app.services.voice_analysis_service.schedule_call_summary_generation"
            ),
            patch(
                "app.services.post_call_email_service.schedule_post_call_email_summary"
            ) as mock_schedule,
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, status
            )

        assert updated is not None
        assert updated.status == status
        mock_schedule.assert_not_called()

    def test_inbound_crm_sync_call_does_not_schedule(self, db, tenant, agent, cs_user):
        """Same skip condition as its sibling hooks: when this session is an
        inbound-CRM-sync call, the email hook must not fire either."""
        flow = _make_flow(
            db,
            tenant,
            agent,
            email_summary_enabled=True,
            summary_to_business_owner_enabled=True,
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
            patch(
                "app.services.voice_analysis_service.schedule_call_summary_generation"
            ),
            patch(
                "app.services.post_call_email_service.schedule_post_call_email_summary"
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
        """Fail-open: an exception raised inside schedule_post_call_email_summary
        must not roll back the status update or propagate to the caller."""
        flow = _make_flow(db, tenant, agent, email_summary_enabled=True)
        session = _make_session(db, tenant, agent, cs_user, call_flow=flow)

        with (
            _naive_utcnow_patch(),
            patch(
                "app.services.voice_analysis_service.schedule_call_summary_generation"
            ),
            patch(
                "app.services.post_call_email_service.schedule_post_call_email_summary",
                side_effect=RuntimeError("simulated ARQ enqueue failure"),
            ),
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        assert updated.status == "completed"
        assert updated.duration is not None


@pytest.mark.usefixtures("db")
class TestPostCallSlackSummaryHook:
    """Coverage for the Slack post-call-summary hook wired into the same
    update_call_session_status() completed-call block as the email hook
    above. Gated on call_flow.slack_summary_enabled AND
    tenant_has_slack_connected(db, tenant_id) — see
    app/services/call_session_service.py's "Post-call Slack summary" block.
    """

    def test_completed_call_with_slack_enabled_and_connected_schedules_job(
        self, db, tenant, agent, cs_user
    ):
        flow = _make_flow(db, tenant, agent, slack_summary_enabled=True)
        session = _make_session(db, tenant, agent, cs_user, call_flow=flow)

        with (
            _naive_utcnow_patch(),
            patch(
                "app.services.voice_analysis_service.schedule_call_summary_generation"
            ),
            patch(
                "app.services.post_call_email_service.schedule_post_call_email_summary"
            ),
            patch(
                "app.services.slack_service.tenant_has_slack_connected",
                return_value=True,
            ),
            patch("app.services.slack_service.schedule_slack_summary") as mock_schedule,
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        assert updated.status == "completed"
        mock_schedule.assert_called_once_with(session.id)

    def test_slack_enabled_but_workspace_not_connected_does_not_schedule(
        self, db, tenant, agent, cs_user
    ):
        """slack_summary_enabled alone is not sufficient — the workspace must
        also have an active Slack connection."""
        flow = _make_flow(db, tenant, agent, slack_summary_enabled=True)
        session = _make_session(db, tenant, agent, cs_user, call_flow=flow)

        with (
            _naive_utcnow_patch(),
            patch(
                "app.services.voice_analysis_service.schedule_call_summary_generation"
            ),
            patch(
                "app.services.post_call_email_service.schedule_post_call_email_summary"
            ),
            patch(
                "app.services.slack_service.tenant_has_slack_connected",
                return_value=False,
            ),
            patch("app.services.slack_service.schedule_slack_summary") as mock_schedule,
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        assert updated.status == "completed"
        mock_schedule.assert_not_called()

    def test_slack_disabled_on_flow_does_not_schedule_even_if_connected(
        self, db, tenant, agent, cs_user
    ):
        flow = _make_flow(db, tenant, agent, slack_summary_enabled=False)
        session = _make_session(db, tenant, agent, cs_user, call_flow=flow)

        with (
            _naive_utcnow_patch(),
            patch(
                "app.services.voice_analysis_service.schedule_call_summary_generation"
            ),
            patch(
                "app.services.post_call_email_service.schedule_post_call_email_summary"
            ),
            patch(
                "app.services.slack_service.tenant_has_slack_connected",
                return_value=True,
            ),
            patch("app.services.slack_service.schedule_slack_summary") as mock_schedule,
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        assert updated.status == "completed"
        mock_schedule.assert_not_called()

    def test_completed_call_with_no_call_flow_does_not_schedule_slack(
        self, db, tenant, agent, cs_user
    ):
        session = _make_session(db, tenant, agent, cs_user, call_flow=None)

        with (
            _naive_utcnow_patch(),
            patch(
                "app.services.voice_analysis_service.schedule_call_summary_generation"
            ),
            patch(
                "app.services.post_call_email_service.schedule_post_call_email_summary"
            ),
            patch("app.services.slack_service.schedule_slack_summary") as mock_schedule,
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        assert updated.status == "completed"
        mock_schedule.assert_not_called()

    @pytest.mark.parametrize("status", ["failed", "no_answer", "ringing"])
    def test_non_completed_status_does_not_schedule_slack(
        self, db, tenant, agent, cs_user, status
    ):
        flow = _make_flow(db, tenant, agent, slack_summary_enabled=True)
        session = _make_session(db, tenant, agent, cs_user, call_flow=flow)

        with (
            _naive_utcnow_patch(),
            patch(
                "app.services.voice_analysis_service.schedule_call_summary_generation"
            ),
            patch(
                "app.services.post_call_email_service.schedule_post_call_email_summary"
            ),
            patch(
                "app.services.slack_service.tenant_has_slack_connected",
                return_value=True,
            ),
            patch("app.services.slack_service.schedule_slack_summary") as mock_schedule,
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, status
            )

        assert updated is not None
        assert updated.status == status
        mock_schedule.assert_not_called()

    def test_inbound_crm_sync_call_does_not_schedule_slack(
        self, db, tenant, agent, cs_user
    ):
        """Same skip condition as the sibling email/HubSpot hooks: an
        inbound-CRM-sync call must not trigger the Slack summary either."""
        flow = _make_flow(db, tenant, agent, slack_summary_enabled=True)
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
            patch(
                "app.services.voice_analysis_service.schedule_call_summary_generation"
            ),
            patch(
                "app.services.post_call_email_service.schedule_post_call_email_summary"
            ),
            patch(
                "app.services.slack_service.tenant_has_slack_connected",
                return_value=True,
            ),
            patch("app.services.slack_service.schedule_slack_summary") as mock_schedule,
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        assert updated.status == "completed"
        mock_schedule.assert_not_called()

    def test_slack_hook_failure_is_swallowed_and_status_update_still_succeeds(
        self, db, tenant, agent, cs_user
    ):
        """Fail-open per app/services/call_session_service.py's Slack hook
        try/except: an exception raised while checking/scheduling the Slack
        summary must never propagate out of update_call_session_status."""
        flow = _make_flow(db, tenant, agent, slack_summary_enabled=True)
        session = _make_session(db, tenant, agent, cs_user, call_flow=flow)

        with (
            _naive_utcnow_patch(),
            patch(
                "app.services.voice_analysis_service.schedule_call_summary_generation"
            ),
            patch(
                "app.services.post_call_email_service.schedule_post_call_email_summary"
            ),
            patch(
                "app.services.slack_service.tenant_has_slack_connected",
                side_effect=RuntimeError("DB connection dropped"),
            ),
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        assert updated.status == "completed"
        assert updated.duration is not None

    def test_schedule_slack_summary_failure_is_swallowed(
        self, db, tenant, agent, cs_user
    ):
        """Fail-open: an exception raised inside schedule_slack_summary itself
        (not just the connectivity check) must also not break call completion."""
        flow = _make_flow(db, tenant, agent, slack_summary_enabled=True)
        session = _make_session(db, tenant, agent, cs_user, call_flow=flow)

        with (
            _naive_utcnow_patch(),
            patch(
                "app.services.voice_analysis_service.schedule_call_summary_generation"
            ),
            patch(
                "app.services.post_call_email_service.schedule_post_call_email_summary"
            ),
            patch(
                "app.services.slack_service.tenant_has_slack_connected",
                return_value=True,
            ),
            patch(
                "app.services.slack_service.schedule_slack_summary",
                side_effect=RuntimeError("thread spawn failed"),
            ),
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        assert updated.status == "completed"
        assert updated.duration is not None

    def test_two_flows_same_tenant_independent_slack_toggle(
        self, db, tenant, agent, cs_user
    ):
        """Agent/call-flow-specific config: one call flow with Slack enabled
        and another with it disabled in the same tenant must behave
        independently."""
        flow_enabled = _make_flow(db, tenant, agent, slack_summary_enabled=True)
        flow_disabled = _make_flow(db, tenant, agent, slack_summary_enabled=False)
        session_enabled = _make_session(
            db, tenant, agent, cs_user, call_flow=flow_enabled
        )
        session_disabled = _make_session(
            db, tenant, agent, cs_user, call_flow=flow_disabled
        )

        with (
            _naive_utcnow_patch(),
            patch(
                "app.services.voice_analysis_service.schedule_call_summary_generation"
            ),
            patch(
                "app.services.post_call_email_service.schedule_post_call_email_summary"
            ),
            patch(
                "app.services.slack_service.tenant_has_slack_connected",
                return_value=True,
            ),
            patch("app.services.slack_service.schedule_slack_summary") as mock_schedule,
        ):
            call_session_service.update_call_session_status(
                db, session_enabled.id, "completed"
            )
            call_session_service.update_call_session_status(
                db, session_disabled.id, "completed"
            )

        mock_schedule.assert_called_once_with(session_enabled.id)
