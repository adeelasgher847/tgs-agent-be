"""Unit tests for app.services.caller_memory_service.

Coverage:
  - get_caller_memory_context_block_for_call: disabled flow / no phone -> ""
  - cache hit returns cached value without re-querying
  - fetch + format + cache on first call
  - fails open on timeout and on unexpected exception
  - _format_caller_memory_block header count and ordering
  - _fetch_recent_summaries query filters (real sqlite session)
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.services import caller_memory_service
from conftest import TestingSessionLocal

_TENANT_ID = uuid.uuid4()
_FLOW_ID = uuid.uuid4()
_SESSION_ID = uuid.uuid4()


@pytest.fixture(autouse=True)
def override_session_local(db):
    class CloseSafeSessionWrapper:
        def __init__(self, session):
            self._session = session

        def __getattr__(self, name):
            return getattr(self._session, name)

        def close(self):
            pass

    with patch("app.services.caller_memory_service.SessionLocal", lambda: CloseSafeSessionWrapper(db)):
        yield


def _call_flow(*, enabled=True, window=3):
    flow = MagicMock()
    flow.id = _FLOW_ID
    flow.caller_memory_enabled = enabled
    flow.caller_memory_window = window
    return flow


def _call_session(*, from_number="+15550001111", metadata=None):
    cs = MagicMock()
    cs.id = _SESSION_ID
    cs.tenant_id = _TENANT_ID
    cs.call_flow_id = _FLOW_ID
    cs.from_number = from_number
    cs.call_metadata = metadata
    return cs


def _past_session(days_ago: int, summary: str):
    s = MagicMock()
    s.start_time = datetime.now(timezone.utc) - timedelta(days=days_ago)
    s.transcript_summary = summary
    return s


class TestGetCallerMemoryContextBlockForCall:
    @pytest.mark.anyio
    async def test_disabled_flow_returns_empty(self):
        db = MagicMock()
        block = await caller_memory_service.get_caller_memory_context_block_for_call(
            db, _call_session(), _call_flow(enabled=False)
        )
        assert block == ""
        db.execute.assert_not_called()

    @pytest.mark.anyio
    async def test_no_from_number_returns_empty(self):
        db = MagicMock()
        block = await caller_memory_service.get_caller_memory_context_block_for_call(
            db, _call_session(from_number=None), _call_flow()
        )
        assert block == ""

    @pytest.mark.anyio
    async def test_no_call_session_returns_empty(self):
        db = MagicMock()
        block = await caller_memory_service.get_caller_memory_context_block_for_call(
            db, None, _call_flow()
        )
        assert block == ""

    @pytest.mark.anyio
    async def test_no_call_flow_returns_empty(self):
        db = MagicMock()
        block = await caller_memory_service.get_caller_memory_context_block_for_call(
            db, _call_session(), None
        )
        assert block == ""

    @pytest.mark.anyio
    async def test_returns_cached_value_without_refetching(self):
        db = MagicMock()
        call_session = _call_session(metadata={"caller_memory_context": "CALLER HISTORY cached"})

        with patch.object(caller_memory_service, "_fetch_recent_summaries") as mock_fetch:
            block = await caller_memory_service.get_caller_memory_context_block_for_call(
                db, call_session, _call_flow()
            )

        mock_fetch.assert_not_called()
        assert block == "CALLER HISTORY cached"

    @pytest.mark.anyio
    async def test_fetches_formats_and_caches_on_first_call(self):
        db = MagicMock()
        call_session = _call_session(metadata={})
        sessions = [
            _past_session(1, "Booked an appointment for Tuesday."),
            _past_session(10, "Asked about pricing."),
        ]

        with patch.object(caller_memory_service, "_fetch_recent_summaries", return_value=sessions):
            block = await caller_memory_service.get_caller_memory_context_block_for_call(
                db, call_session, _call_flow(window=3)
            )

        assert block.startswith("<caller_history>\nCALLER HISTORY (last 2 interactions):")
        assert "Booked an appointment for Tuesday." in block
        assert "Asked about pricing." in block
        assert block.endswith("End of caller history.\n</caller_history>")
        assert call_session.call_metadata["caller_memory_context"] == block
        db.flush.assert_called_once()
        db.commit.assert_not_called()

    @pytest.mark.anyio
    async def test_no_history_returns_empty_and_caches_empty_string(self):
        db = MagicMock()
        call_session = _call_session(metadata={})

        with patch.object(caller_memory_service, "_fetch_recent_summaries", return_value=[]):
            block = await caller_memory_service.get_caller_memory_context_block_for_call(
                db, call_session, _call_flow()
            )

        assert block == ""
        assert call_session.call_metadata["caller_memory_context"] == ""

    @pytest.mark.anyio
    async def test_fails_open_on_timeout(self):
        db = MagicMock()
        call_session = _call_session(metadata={})

        def _slow_fetch(*args, **kwargs):
            import time

            time.sleep(0.05)
            return []

        with patch.object(caller_memory_service, "_fetch_recent_summaries", side_effect=_slow_fetch):
            with patch.object(
                caller_memory_service, "_DEFAULT_FETCH_TIMEOUT_SEC", 0.01
            ):
                block = await caller_memory_service.get_caller_memory_context_block_for_call(
                    db, call_session, _call_flow()
                )

        assert block == ""
        # Still caches the fail-open empty block so subsequent turns don't retry.
        assert call_session.call_metadata["caller_memory_context"] == ""

    @pytest.mark.anyio
    async def test_fails_open_on_exception(self):
        db = MagicMock()
        call_session = _call_session(metadata={})

        with patch.object(
            caller_memory_service, "_fetch_recent_summaries", side_effect=RuntimeError("db down")
        ):
            block = await caller_memory_service.get_caller_memory_context_block_for_call(
                db, call_session, _call_flow()
            )

        assert block == ""

    @pytest.mark.anyio
    async def test_fails_open_on_flush_error(self):
        db = MagicMock()
        db.flush.side_effect = Exception("flush failed")
        nested_mock = MagicMock()
        db.begin_nested.return_value = nested_mock
        nested_mock.__enter__ = MagicMock()
        nested_mock.__exit__ = MagicMock()

        call_session = _call_session(metadata={})
        sessions = [_past_session(1, "Summary text")]

        with patch.object(caller_memory_service, "_fetch_recent_summaries", return_value=sessions):
            block = await caller_memory_service.get_caller_memory_context_block_for_call(
                db, call_session, _call_flow()
            )

        assert "Summary text" in block
        db.flush.assert_called_once()


class TestFormatCallerMemoryBlock:
    def test_empty_sessions_returns_empty_string(self):
        assert caller_memory_service._format_caller_memory_block([]) == ""

    def test_formats_header_with_actual_count_and_dates(self):
        sessions = [
            _past_session(0, "First summary"),
        ]
        block = caller_memory_service._format_caller_memory_block(sessions)
        lines = block.split("\n")
        assert lines[0] == "<caller_history>"
        assert lines[1] == "CALLER HISTORY (last 1 interactions):"
        assert lines[2].startswith("- Call on ")
        assert "First summary" in lines[2]
        assert lines[-2] == "End of caller history."
        assert lines[-1] == "</caller_history>"


class TestSanitizeSummary:
    def test_replaces_control_characters(self):
        raw = "Hello\x00World\x1f!\x7f"
        sanitized = caller_memory_service._sanitize_summary(raw)
        assert sanitized == "Hello World !"

    def test_truncates_long_summaries(self):
        raw = "a" * 500
        sanitized = caller_memory_service._sanitize_summary(raw)
        assert len(sanitized) == 401  # 400 chars + 1 ellipsis char
        assert sanitized.endswith("…")
        assert sanitized[:-1] == "a" * 400


class TestFetchRecentSummariesQuery:
    """Exercises the real SQL query against the shared sqlite test database."""

    def test_filters_by_tenant_flow_number_status_and_orders_desc(self, db):
        from app.models.agent import Agent
        from app.models.call_flow import CallFlow
        from app.models.tenant import Tenant
        from app.models.user import User
        from app.models.call_session import CallSession

        tenant = Tenant(name=f"CM-{uuid.uuid4().hex[:6]}", schema_name=f"cm_{uuid.uuid4().hex[:6]}")
        other_tenant = Tenant(
            name=f"CM-other-{uuid.uuid4().hex[:6]}", schema_name=f"cm_other_{uuid.uuid4().hex[:6]}"
        )
        db.add_all([tenant, other_tenant])
        db.flush()

        agent = Agent(
            tenant_id=tenant.id,
            name="Memory Test Agent",
            status="active",
            llm_model="gpt-4o-mini",
            tts_provider_slug="elevenlabs",
            tts_voice_external_id="voice-x",
            tts_language="en",
        )
        db.add(agent)
        db.flush()

        user = User(
            email=f"cm-{uuid.uuid4().hex[:6]}@example.com",
            first_name="CM",
            last_name="User",
            hashed_password="",
            current_tenant_id=tenant.id,
        )
        db.add(user)
        db.flush()

        flow = CallFlow(tenant_id=tenant.id, agent_id=agent.id, name="Memory Flow", direction="inbound")
        db.add(flow)
        db.flush()

        current_session = CallSession(
            user_id=user.id,
            agent_id=agent.id,
            tenant_id=tenant.id,
            call_flow_id=flow.id,
            from_number="+15551234567",
            status="active",
            start_time=datetime.now(timezone.utc),
        )
        db.add(current_session)
        db.flush()

        def _completed_call(*, days_ago, from_number, summary, tenant_id, status="completed"):
            return CallSession(
                user_id=user.id,
                agent_id=agent.id,
                tenant_id=tenant_id,
                call_flow_id=flow.id,
                from_number=from_number,
                status=status,
                transcript_summary=summary,
                start_time=datetime.now(timezone.utc) - timedelta(days=days_ago),
            )

        matching_recent = _completed_call(
            days_ago=1, from_number="+15551234567", summary="Most recent call", tenant_id=tenant.id
        )
        matching_older = _completed_call(
            days_ago=5, from_number="+15551234567", summary="Older call", tenant_id=tenant.id
        )
        wrong_number = _completed_call(
            days_ago=1, from_number="+19998887777", summary="Wrong number", tenant_id=tenant.id
        )
        not_completed = _completed_call(
            days_ago=1,
            from_number="+15551234567",
            summary="Still active",
            tenant_id=tenant.id,
            status="active",
        )
        no_summary = _completed_call(
            days_ago=1, from_number="+15551234567", summary=None, tenant_id=tenant.id
        )
        other_tenant_call = _completed_call(
            days_ago=1, from_number="+15551234567", summary="Other tenant", tenant_id=other_tenant.id
        )
        db.add_all(
            [matching_recent, matching_older, wrong_number, not_completed, no_summary, other_tenant_call]
        )
        db.commit()

        results = caller_memory_service._fetch_recent_summaries(
            current_session.tenant_id,
            current_session.call_flow_id,
            current_session.from_number,
            current_session.id,
            window=10
        )

        assert [r.transcript_summary for r in results] == ["Most recent call", "Older call"]

    def test_respects_window_limit(self, db):
        from app.models.agent import Agent
        from app.models.call_flow import CallFlow
        from app.models.tenant import Tenant
        from app.models.user import User
        from app.models.call_session import CallSession

        tenant = Tenant(name=f"CMW-{uuid.uuid4().hex[:6]}", schema_name=f"cmw_{uuid.uuid4().hex[:6]}")
        db.add(tenant)
        db.flush()

        agent = Agent(
            tenant_id=tenant.id,
            name="Memory Window Agent",
            status="active",
            llm_model="gpt-4o-mini",
            tts_provider_slug="elevenlabs",
            tts_voice_external_id="voice-x",
            tts_language="en",
        )
        db.add(agent)
        db.flush()

        user = User(
            email=f"cmw-{uuid.uuid4().hex[:6]}@example.com",
            first_name="CMW",
            last_name="User",
            hashed_password="",
            current_tenant_id=tenant.id,
        )
        db.add(user)
        db.flush()

        flow = CallFlow(tenant_id=tenant.id, agent_id=agent.id, name="Memory Window Flow", direction="inbound")
        db.add(flow)
        db.flush()

        current_session = CallSession(
            user_id=user.id,
            agent_id=agent.id,
            tenant_id=tenant.id,
            call_flow_id=flow.id,
            from_number="+15559990000",
            status="active",
            start_time=datetime.now(timezone.utc),
        )
        db.add(current_session)
        db.flush()

        for i in range(5):
            db.add(
                CallSession(
                    user_id=user.id,
                    agent_id=agent.id,
                    tenant_id=tenant.id,
                    call_flow_id=flow.id,
                    from_number="+15559990000",
                    status="completed",
                    transcript_summary=f"Call {i}",
                    start_time=datetime.now(timezone.utc) - timedelta(days=i + 1),
                )
            )
        db.commit()

        results = caller_memory_service._fetch_recent_summaries(
            current_session.tenant_id,
            current_session.call_flow_id,
            current_session.from_number,
            current_session.id,
            window=2
        )
        assert len(results) == 2
