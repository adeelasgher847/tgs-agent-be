"""Unit tests for the demo-link minute-accounting hook wired into
CallSessionService.update_call_session_status() / _record_demo_link_usage().

See app/routers/sdk.py::demo_call_token for how call_metadata["demo_link"]
gets stashed on a CallSession created via the public demo-link flow.
"""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch

import pytest

from app.models.agent import Agent
from app.models.call_flow_demo_link import CallFlowDemoLink
from app.models.call_flow_demo_link_visitor_usage import CallFlowDemoLinkVisitorUsage
from app.models.call_session import CallSession
from app.models.tenant import Tenant
from app.models.user import User
from app.services.call_session_service import call_session_service


@contextmanager
def _naive_utcnow_patch():
    """Work around a SQLite-only testing artifact: SQLAlchemy's
    DateTime(timezone=True) column does not round-trip tzinfo through SQLite
    (unlike the real Postgres column), so a CallSession reloaded from the test
    DB always has a naive start_time. update_call_session_status() then computes
    ``datetime.now(timezone.utc) - call_session.start_time``, which raises
    "can't subtract offset-naive and offset-aware datetimes" purely because of
    this SQLite limitation -- never against the real Postgres schema. Patch
    datetime.now(tz) inside call_session_service to also return a naive value
    so the real duration-calculation code path can still be exercised here.
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
        name=f"DemoUsageWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"demo_usage_ws_{uuid.uuid4().hex[:8]}",
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
        name="Demo Usage Agent",
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
        email=f"cs-user-{uuid.uuid4().hex[:8]}@example.com",
        first_name="CS",
        last_name="User",
        hashed_password="",
        current_tenant_id=tenant.id,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


@pytest.fixture
def demo_link(db, tenant: Tenant, agent: Agent) -> CallFlowDemoLink:
    from app.models.call_flow import CallFlow

    flow = CallFlow(
        tenant_id=tenant.id,
        agent_id=agent.id,
        name="Usage Flow",
        direction="inbound",
    )
    db.add(flow)
    db.commit()
    db.refresh(flow)

    link = CallFlowDemoLink(
        tenant_id=tenant.id,
        call_flow_id=flow.id,
        token=uuid.uuid4().hex,
        total_minutes_used=Decimal("0"),
    )
    db.add(link)
    db.commit()
    db.refresh(link)
    return link


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
class TestRecordDemoLinkUsage:
    def test_completed_call_credits_link_and_visitor_minutes(
        self, db, tenant, agent, cs_user, demo_link
    ):
        visitor_usage = CallFlowDemoLinkVisitorUsage(
            demo_link_id=demo_link.id,
            visitor_id="visitor-usage-1",
            minutes_used=Decimal("0"),
        )
        db.add(visitor_usage)
        db.commit()

        session = _make_session(
            db, tenant, agent, cs_user,
            call_metadata={
                "demo_link": {
                    "demo_link_id": str(demo_link.id),
                    "visitor_id": "visitor-usage-1",
                }
            },
            minutes_ago=5,
        )

        with _naive_utcnow_patch():
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )
        assert updated is not None
        assert updated.duration is not None
        expected_minutes = Decimal(updated.duration) / Decimal(60)

        db.refresh(demo_link)
        db.refresh(visitor_usage)
        assert demo_link.total_minutes_used == expected_minutes
        assert visitor_usage.minutes_used == expected_minutes

    def test_no_demo_link_key_is_noop(self, db, tenant, agent, cs_user):
        """call_metadata has no 'demo_link' key — the overwhelming majority
        of calls. Must not raise and must not touch any demo-link tables."""
        session = _make_session(
            db, tenant, agent, cs_user,
            call_metadata={"some_other_key": "value"},
        )

        with _naive_utcnow_patch():
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )
        assert updated is not None
        assert updated.status == "completed"

    def test_null_call_metadata_is_noop(self, db, tenant, agent, cs_user):
        session = _make_session(db, tenant, agent, cs_user, call_metadata=None)

        with _naive_utcnow_patch():
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )
        assert updated is not None
        assert updated.status == "completed"

    def test_demo_link_pointing_at_deleted_row_fails_open(
        self, db, tenant, agent, cs_user
    ):
        """demo_link_id references a row that no longer exists (e.g. the demo
        link was deleted between call start and call end). Must not raise and
        must not block the rest of update_call_session_status."""
        nonexistent_id = str(uuid.uuid4())
        session = _make_session(
            db, tenant, agent, cs_user,
            call_metadata={
                "demo_link": {
                    "demo_link_id": nonexistent_id,
                    "visitor_id": "visitor-orphaned",
                }
            },
        )

        with _naive_utcnow_patch():
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )
        assert updated is not None
        assert updated.status == "completed"
        assert updated.duration is not None

    def test_lookup_exception_fails_open_and_does_not_block_status_update(
        self, db, tenant, agent, cs_user, demo_link
    ):
        """Simulate an unexpected failure inside _record_demo_link_usage
        (e.g. a DB error during the demo-link lookup). The outer try/except
        in update_call_session_status must swallow it, log, and still return
        the updated call session."""
        session = _make_session(
            db, tenant, agent, cs_user,
            call_metadata={
                "demo_link": {
                    "demo_link_id": str(demo_link.id),
                    "visitor_id": "visitor-boom",
                }
            },
        )

        with _naive_utcnow_patch(), patch.object(
            call_session_service,
            "_record_demo_link_usage",
            side_effect=RuntimeError("simulated demo-link lookup failure"),
        ):
            updated = call_session_service.update_call_session_status(
                db, session.id, "completed"
            )

        assert updated is not None
        assert updated.status == "completed"
        assert updated.duration is not None

        # Demo link itself must be untouched since accounting never ran
        db.refresh(demo_link)
        assert demo_link.total_minutes_used == Decimal("0")

    def test_missing_duration_skips_accounting(self, db, tenant, agent, cs_user, demo_link):
        """No duration computed on the call session — accounting must not run
        and must not raise. Exercised directly against _record_demo_link_usage
        since CallSession.start_time is NOT NULL, so a real "no duration" row
        can't be persisted end-to-end via update_call_session_status."""
        session = _make_session(
            db, tenant, agent, cs_user,
            call_metadata={
                "demo_link": {
                    "demo_link_id": str(demo_link.id),
                    "visitor_id": "visitor-no-duration",
                }
            },
        )
        session.duration = None

        call_session_service._record_demo_link_usage(db, session)
        assert session.duration is None

        db.refresh(demo_link)
        assert demo_link.total_minutes_used == Decimal("0")

    def test_completed_demo_link_call_skips_crm_writeback(self, db, tenant, agent, cs_user, demo_link):
        """
        A demo-link CallSession has an anonymous customer_phone_number
        placeholder, not a real contact — CRM write-back scheduling must be
        skipped entirely for it, even when the tenant has HubSpot/Salesforce/
        GHL connected, rather than relying on the placeholder phone number
        being unmatchable as an implicit safety net.
        """
        session = _make_session(
            db, tenant, agent, cs_user,
            call_metadata={
                "demo_link": {
                    "demo_link_id": str(demo_link.id),
                    "visitor_id": "visitor-crm-skip",
                }
            },
            minutes_ago=5,
        )

        with _naive_utcnow_patch(), \
             patch("app.services.hubspot_service.tenant_has_hubspot_connected", return_value=True) as mock_hs_conn, \
             patch("app.services.hubspot_service.schedule_hubspot_writeback") as mock_hs_sched, \
             patch("app.services.salesforce_service.tenant_has_salesforce_connected", return_value=True) as mock_sf_conn, \
             patch("app.services.salesforce_service.schedule_salesforce_writeback") as mock_sf_sched, \
             patch("app.services.ghl_service.tenant_has_ghl_connected", return_value=True) as mock_ghl_conn, \
             patch("app.services.ghl_service.schedule_ghl_writeback") as mock_ghl_sched:
            call_session_service.update_call_session_status(db, session.id, "completed")

        mock_hs_conn.assert_not_called()
        mock_hs_sched.assert_not_called()
        mock_sf_conn.assert_not_called()
        mock_sf_sched.assert_not_called()
        mock_ghl_conn.assert_not_called()
        mock_ghl_sched.assert_not_called()

    def test_completed_non_demo_link_call_still_triggers_crm_writeback(self, db, tenant, agent, cs_user):
        """Sanity check the skip is scoped to demo-link calls only — an
        ordinary completed call must still trigger CRM write-back scheduling
        when the tenant has an integration connected."""
        session = _make_session(
            db, tenant, agent, cs_user,
            call_metadata=None,
            minutes_ago=5,
        )

        with _naive_utcnow_patch(), \
             patch("app.services.hubspot_service.tenant_has_hubspot_connected", return_value=True), \
             patch("app.services.hubspot_service.schedule_hubspot_writeback") as mock_hs_sched, \
             patch("app.services.salesforce_service.tenant_has_salesforce_connected", return_value=False), \
             patch("app.services.ghl_service.tenant_has_ghl_connected", return_value=False):
            call_session_service.update_call_session_status(db, session.id, "completed")

        mock_hs_sched.assert_called_once_with(session.id)
