"""Unit tests for the System Webhooks hooks wired into
CallSessionService.update_call_session_status():

  - Status Webhook "end" event, fired on any terminal status transition
    (completed/failed/busy/no_answer) when call_flow.status_webhook_enabled.
  - Post-Call Webhook scheduling, fired only on "completed" (mirrors the
    sibling Slack/email/analysis hooks' gating) when
    call_flow.post_call_webhook_url is set.

Mirrors tests/services/test_call_session_post_call_analysis_hook.py's
fixture setup and mocking conventions exactly, since both hooks live in the
same try/except-fail-open blocks inside the same method.
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
        name=f"SysWebhookHookWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"sys_webhook_hook_ws_{uuid.uuid4().hex[:8]}",
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
        name="System Webhook Hook Agent",
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
        email=f"sys-webhook-hook-user-{uuid.uuid4().hex[:8]}@example.com",
        first_name="SysWebhook",
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
    status_webhook_enabled: bool = False,
    status_webhook_url: str | None = None,
    post_call_webhook_url: str | None = None,
) -> CallFlow:
    f = CallFlow(
        tenant_id=tenant.id,
        agent_id=agent.id,
        name="System Webhook Hook Flow",
        direction="inbound",
        status_webhook_enabled=status_webhook_enabled,
        status_webhook_url=status_webhook_url,
        post_call_webhook_url=post_call_webhook_url,
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


_ALL_SIBLING_HOOK_PATCHES = (
    "app.services.voice_analysis_service.schedule_call_summary_generation",
    "app.services.post_call_email_service.schedule_post_call_email_summary",
    "app.services.post_call_analysis_service.schedule_run_post_call_analysis",
)


@contextmanager
def _patch_sibling_hooks():
    """Silence the other schedule_* hooks in update_call_session_status so
    they don't interfere with (or get accidentally exercised as) assertions
    about the two System Webhooks hooks under test here."""
    with (
        patch(_ALL_SIBLING_HOOK_PATCHES[0]),
        patch(_ALL_SIBLING_HOOK_PATCHES[1]),
        patch(_ALL_SIBLING_HOOK_PATCHES[2]),
    ):
        yield


@pytest.mark.usefixtures("db")
class TestStatusWebhookEndEventHook:
    @pytest.mark.parametrize("status", ["completed", "failed", "busy", "no_answer"])
    def test_terminal_status_with_status_webhook_enabled_schedules_end_event(
        self, db, tenant, agent, cs_user, status
    ):
        flow = _make_flow(
            db,
            tenant,
            agent,
            status_webhook_enabled=True,
            status_webhook_url="https://example.com/status",
        )
        session = _make_session(db, tenant, agent, cs_user, call_flow=flow)

        with (
            _naive_utcnow_patch(),
            _patch_sibling_hooks(),
            patch(
                "app.services.system_webhook_service.schedule_status_webhook"
            ) as mock_schedule,
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, status
            )

        assert updated is not None
        assert updated.status == status
        mock_schedule.assert_called_once_with(session.id, "call.ended")

    def test_status_webhook_disabled_does_not_schedule(
        self, db, tenant, agent, cs_user
    ):
        flow = _make_flow(
            db,
            tenant,
            agent,
            status_webhook_enabled=False,
            status_webhook_url="https://example.com/status",
        )
        session = _make_session(db, tenant, agent, cs_user, call_flow=flow)

        with (
            _naive_utcnow_patch(),
            _patch_sibling_hooks(),
            patch(
                "app.services.system_webhook_service.schedule_status_webhook"
            ) as mock_schedule,
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        mock_schedule.assert_not_called()

    def test_no_call_flow_does_not_schedule(self, db, tenant, agent, cs_user):
        session = _make_session(db, tenant, agent, cs_user, call_flow=None)

        with (
            _naive_utcnow_patch(),
            _patch_sibling_hooks(),
            patch(
                "app.services.system_webhook_service.schedule_status_webhook"
            ) as mock_schedule,
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        mock_schedule.assert_not_called()

    def test_non_terminal_status_does_not_schedule(self, db, tenant, agent, cs_user):
        flow = _make_flow(
            db,
            tenant,
            agent,
            status_webhook_enabled=True,
            status_webhook_url="https://example.com/status",
        )
        session = _make_session(db, tenant, agent, cs_user, call_flow=flow)

        with (
            _naive_utcnow_patch(),
            _patch_sibling_hooks(),
            patch(
                "app.services.system_webhook_service.schedule_status_webhook"
            ) as mock_schedule,
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "ringing"
            )

        assert updated is not None
        mock_schedule.assert_not_called()

    def test_schedule_failure_is_swallowed_and_status_update_still_succeeds(
        self, db, tenant, agent, cs_user
    ):
        flow = _make_flow(
            db,
            tenant,
            agent,
            status_webhook_enabled=True,
            status_webhook_url="https://example.com/status",
        )
        session = _make_session(db, tenant, agent, cs_user, call_flow=flow)

        with (
            _naive_utcnow_patch(),
            _patch_sibling_hooks(),
            patch(
                "app.services.system_webhook_service.schedule_status_webhook",
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
class TestPostCallWebhookScheduleHook:
    def test_completed_with_url_configured_schedules_job(
        self, db, tenant, agent, cs_user
    ):
        flow = _make_flow(
            db, tenant, agent, post_call_webhook_url="https://example.com/post-call"
        )
        session = _make_session(db, tenant, agent, cs_user, call_flow=flow)

        with (
            _naive_utcnow_patch(),
            _patch_sibling_hooks(),
            patch(
                "app.services.system_webhook_service.schedule_post_call_webhook"
            ) as mock_schedule,
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        assert updated.status == "completed"
        mock_schedule.assert_called_once_with(session.id)

    def test_completed_with_no_url_does_not_schedule(self, db, tenant, agent, cs_user):
        flow = _make_flow(db, tenant, agent, post_call_webhook_url=None)
        session = _make_session(db, tenant, agent, cs_user, call_flow=flow)

        with (
            _naive_utcnow_patch(),
            _patch_sibling_hooks(),
            patch(
                "app.services.system_webhook_service.schedule_post_call_webhook"
            ) as mock_schedule,
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        mock_schedule.assert_not_called()

    def test_completed_with_no_call_flow_does_not_schedule(
        self, db, tenant, agent, cs_user
    ):
        session = _make_session(db, tenant, agent, cs_user, call_flow=None)

        with (
            _naive_utcnow_patch(),
            _patch_sibling_hooks(),
            patch(
                "app.services.system_webhook_service.schedule_post_call_webhook"
            ) as mock_schedule,
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        mock_schedule.assert_not_called()

    @pytest.mark.parametrize("status", ["failed", "no_answer", "busy", "ringing"])
    def test_non_completed_status_does_not_schedule(
        self, db, tenant, agent, cs_user, status
    ):
        """Documented asymmetry vs. the status-webhook 'end' event (which
        fires on ALL terminal statuses): the post-call webhook only fires on
        'completed', mirroring the Slack/email/analysis hooks' gating."""
        flow = _make_flow(
            db, tenant, agent, post_call_webhook_url="https://example.com/post-call"
        )
        session = _make_session(db, tenant, agent, cs_user, call_flow=flow)

        with (
            _naive_utcnow_patch(),
            _patch_sibling_hooks(),
            patch(
                "app.services.system_webhook_service.schedule_post_call_webhook"
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
        inbound-CRM-sync call, the post-call-webhook hook must not fire either."""
        flow = _make_flow(
            db, tenant, agent, post_call_webhook_url="https://example.com/post-call"
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
            _patch_sibling_hooks(),
            patch(
                "app.services.system_webhook_service.schedule_post_call_webhook"
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
        flow = _make_flow(
            db, tenant, agent, post_call_webhook_url="https://example.com/post-call"
        )
        session = _make_session(db, tenant, agent, cs_user, call_flow=flow)

        with (
            _naive_utcnow_patch(),
            _patch_sibling_hooks(),
            patch(
                "app.services.system_webhook_service.schedule_post_call_webhook",
                side_effect=RuntimeError("simulated ARQ enqueue failure"),
            ),
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        assert updated.status == "completed"
        assert updated.duration is not None

    def test_both_hooks_fire_together_on_completed_when_both_configured(
        self, db, tenant, agent, cs_user
    ):
        flow = _make_flow(
            db,
            tenant,
            agent,
            status_webhook_enabled=True,
            status_webhook_url="https://example.com/status",
            post_call_webhook_url="https://example.com/post-call",
        )
        session = _make_session(db, tenant, agent, cs_user, call_flow=flow)

        with (
            _naive_utcnow_patch(),
            _patch_sibling_hooks(),
            patch(
                "app.services.system_webhook_service.schedule_status_webhook"
            ) as mock_status,
            patch(
                "app.services.system_webhook_service.schedule_post_call_webhook"
            ) as mock_post_call,
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        mock_status.assert_called_once_with(session.id, "call.ended")
        mock_post_call.assert_called_once_with(session.id)
