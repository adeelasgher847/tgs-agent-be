"""Tests for GET /api/v2/calls/{call_id}/memory-context.

Coverage:
  - Success: returns the exact cached caller_memory_context string
  - Empty cache: returns "" when caller_memory_context is not cached
  - Workspace isolation: a call belonging to another tenant returns 404
  - Missing session: an unknown call_id returns 404
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.exception_handlers import register_exception_handlers


def _build_app(db_override, principal):
    from app.api.deps import get_db, require_tenant
    from app.api.v2.routers.callback_scheduler import calls_router

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(calls_router)

    mini.dependency_overrides[require_tenant] = lambda: principal
    mini.dependency_overrides[get_db] = lambda: db_override

    return TestClient(mini, raise_server_exceptions=False)


def _principal(tenant_id: uuid.UUID) -> MagicMock:
    principal = MagicMock()
    principal.id = uuid.uuid4()
    principal.current_tenant_id = tenant_id
    return principal


@pytest.fixture
def workspace(db):
    from app.models.tenant import Tenant

    tenant = Tenant(
        name=f"MemDebugWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"mem_debug_ws_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    db.add(tenant)
    db.commit()
    db.refresh(tenant)
    return tenant


@pytest.fixture
def agent(db, workspace):
    from app.models.agent import Agent

    a = Agent(
        tenant_id=workspace.id,
        name="Memory Debug Test Agent",
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
def user(db, workspace):
    from app.models.user import User

    u = User(
        email=f"memdebug-{uuid.uuid4().hex[:6]}@example.com",
        first_name="Mem",
        last_name="Debug",
        hashed_password="",
        current_tenant_id=workspace.id,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _make_call_session(db, *, tenant_id, agent_id, user_id, call_metadata=None):
    from app.models.call_session import CallSession

    cs = CallSession(
        user_id=user_id,
        agent_id=agent_id,
        tenant_id=tenant_id,
        status="completed",
        start_time=datetime.now(timezone.utc),
        call_metadata=call_metadata,
    )
    db.add(cs)
    db.commit()
    db.refresh(cs)
    return cs


@pytest.mark.usefixtures("db")
class TestGetCallMemoryContext:
    def test_returns_exact_cached_context_string(self, db, workspace, agent, user):
        cached_block = (
            "CALLER HISTORY (last 2 interactions):\n"
            "- Call on 2026-06-20: Booked an appointment.\n"
            "- Call on 2026-06-25: Asked about pricing.\n"
            "End of caller history."
        )
        call_session = _make_call_session(
            db,
            tenant_id=workspace.id,
            agent_id=agent.id,
            user_id=user.id,
            call_metadata={"caller_memory_context": cached_block},
        )

        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.get(f"/calls/{call_session.id}/memory-context")

        assert resp.status_code == 200, resp.text
        assert resp.json() == {"memory_context": cached_block}

    def test_empty_cache_returns_empty_string(self, db, workspace, agent, user):
        call_session = _make_call_session(
            db,
            tenant_id=workspace.id,
            agent_id=agent.id,
            user_id=user.id,
            call_metadata={"llm_call_analysis": {"analysis": {}}},
        )

        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.get(f"/calls/{call_session.id}/memory-context")

        assert resp.status_code == 200, resp.text
        assert resp.json() == {"memory_context": ""}

    def test_no_call_metadata_at_all_returns_empty_string(self, db, workspace, agent, user):
        call_session = _make_call_session(
            db,
            tenant_id=workspace.id,
            agent_id=agent.id,
            user_id=user.id,
            call_metadata=None,
        )

        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.get(f"/calls/{call_session.id}/memory-context")

        assert resp.status_code == 200, resp.text
        assert resp.json() == {"memory_context": ""}

    def test_cross_tenant_access_returns_404(self, db, workspace, agent, user):
        from app.models.tenant import Tenant

        other_tenant = Tenant(
            name=f"OtherMemWS-{uuid.uuid4().hex[:8]}",
            schema_name=f"other_mem_ws_{uuid.uuid4().hex[:8]}",
            status="active",
        )
        db.add(other_tenant)
        db.commit()
        db.refresh(other_tenant)

        call_session = _make_call_session(
            db,
            tenant_id=workspace.id,
            agent_id=agent.id,
            user_id=user.id,
            call_metadata={"caller_memory_context": "Should not be visible"},
        )

        # Principal belongs to a different tenant than the call session.
        principal = _principal(other_tenant.id)
        client = _build_app(db, principal)

        resp = client.get(f"/calls/{call_session.id}/memory-context")

        assert resp.status_code == 404

    def test_unknown_call_id_returns_404(self, db, workspace):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.get(f"/calls/{uuid.uuid4()}/memory-context")

        assert resp.status_code == 404
