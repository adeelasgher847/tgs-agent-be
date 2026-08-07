"""
Tests for GET /api/v1/dashboard — the single-response dashboard summary.

Covers:
  - 200 happy path with real data (monthly metrics, credits, recent_* lists)
  - Empty-tenant case (zero everything, still 200 with empty lists / 0s)
  - Tenant isolation (tenant A never sees tenant B's rows/counts)
  - limit=4 enforcement + most-recent-first ordering for recent_* lists
  - 403 when no tenant selected
  - month-scoping of monthly_calls / monthly_spent (CallSession.start_time)
  - response envelope shape (SuccessResponse)
"""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.main import app
from app.models.agent import Agent
from app.models.call_flow import CallFlow
from app.models.call_session import CallSession
from app.models.kb_file import KbFile
from app.models.knowledge_base_document import KnowledgeBase
from app.models.tenant import Tenant
from app.models.user import User


# ── Auth bypass ───────────────────────────────────────────────────────────────

def _mock_user(tenant_id: uuid.UUID | None):
    user = MagicMock()
    user.current_tenant_id = tenant_id
    return user


@contextmanager
def _auth_ctx(tenant_id: uuid.UUID | None):
    from app.api.deps import require_readonly_or_api_key
    import app.middleware.api_key_middleware as auth_mw

    mock_user = _mock_user(tenant_id)
    app.dependency_overrides[require_readonly_or_api_key] = lambda: mock_user
    with patch.object(auth_mw, "_try_jwt_auth", new=AsyncMock(return_value=True)):
        try:
            yield
        finally:
            app.dependency_overrides.pop(require_readonly_or_api_key, None)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tenant(db, *, credits=0) -> Tenant:
    tenant = Tenant(
        id=uuid.uuid4(),
        name=f"Dashboard Tenant {uuid.uuid4().hex[:6]}",
        schema_name=f"dashboard_schema_{uuid.uuid4().hex[:6]}",
        credits=credits,
    )
    db.add(tenant)
    db.commit()
    db.refresh(tenant)
    return tenant


def _make_agent(db, tenant_id) -> Agent:
    agent = Agent(id=uuid.uuid4(), tenant_id=tenant_id, name="Dashboard Agent", is_deleted=False)
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return agent


def _make_call_flow(db, tenant_id, agent_id, *, name, created_at, is_deleted=False) -> CallFlow:
    flow = CallFlow(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        agent_id=agent_id,
        name=name,
        direction="inbound",
        is_deleted=is_deleted,
        created_at=created_at,
    )
    db.add(flow)
    db.commit()
    db.refresh(flow)
    return flow


def _make_kb(db, tenant_id, *, name, created_at) -> KnowledgeBase:
    kb = KnowledgeBase(id=uuid.uuid4(), workspace_id=tenant_id, name=name, created_at=created_at)
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return kb


def _make_kb_file(db, kb_id, *, status="ready") -> KbFile:
    f = KbFile(
        id=uuid.uuid4(),
        kb_id=kb_id,
        original_filename="doc.pdf",
        file_type="pdf",
        status=status,
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    return f


def _make_call_session(
    db,
    tenant_id,
    user_id,
    agent_id,
    *,
    start_time,
    created_at=None,
    cost=0.0,
    status="completed",
    transcript_summary="Caller asked about pricing.",
) -> CallSession:
    call = CallSession(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        user_id=user_id,
        agent_id=agent_id,
        start_time=start_time,
        created_at=created_at or start_time,
        cost=cost,
        status=status,
        call_type="inbound",
        duration=60,
        transcript_summary=transcript_summary,
    )
    db.add(call)
    db.commit()
    db.refresh(call)
    return call


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def any_user_id(db) -> uuid.UUID:
    return db.query(User).first().id


# ── 1. Happy path with real data ───────────────────────────────────────────────

def test_dashboard_happy_path_with_real_data(client, db, any_user_id):
    tenant = _make_tenant(db, credits="42.5000")
    agent = _make_agent(db, tenant.id)
    now = datetime.now(timezone.utc)

    # Call flows: 2 active (varying created_at) + 1 soft-deleted (must be excluded)
    flow1 = _make_call_flow(db, tenant.id, agent.id, name="Flow 1", created_at=now - timedelta(minutes=10))
    flow2 = _make_call_flow(db, tenant.id, agent.id, name="Flow 2", created_at=now - timedelta(minutes=5))
    deleted_flow = _make_call_flow(
        db, tenant.id, agent.id, name="Deleted Flow", created_at=now - timedelta(minutes=1), is_deleted=True
    )

    # Knowledge bases with mixed-status files
    kb1 = _make_kb(db, tenant.id, name="KB 1", created_at=now - timedelta(minutes=10))
    kb2 = _make_kb(db, tenant.id, name="KB 2", created_at=now - timedelta(minutes=5))
    _make_kb_file(db, kb1.id, status="ready")
    _make_kb_file(db, kb1.id, status="ready")
    _make_kb_file(db, kb1.id, status="processing")  # must be excluded from count
    _make_kb_file(db, kb2.id, status="error")  # must be excluded from count

    # Call sessions this month — one with a None summary
    call1 = _make_call_session(
        db, tenant.id, any_user_id, agent.id,
        start_time=now - timedelta(minutes=10), cost=1.5, status="completed",
        transcript_summary="Summary A",
    )
    call2 = _make_call_session(
        db, tenant.id, any_user_id, agent.id,
        start_time=now - timedelta(minutes=5), cost=2.25, status="failed",
        transcript_summary=None,
    )

    with _auth_ctx(tenant.id):
        resp = client.get("/api/v1/dashboard")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    data = body["data"]

    assert data["monthly_calls"] == 2
    assert data["monthly_spent"] == pytest.approx(3.75)
    assert data["credits"] == pytest.approx(42.5)

    flow_ids = [f["id"] for f in data["recent_call_flows"]]
    assert str(deleted_flow.id) not in flow_ids
    assert flow_ids == [str(flow2.id), str(flow1.id)]  # most recent first
    assert len(data["recent_call_flows"]) <= 4

    kb_items = {k["id"]: k for k in data["recent_knowledge_bases"]}
    assert kb_items[str(kb1.id)]["file_count"] == 2
    assert kb_items[str(kb2.id)]["file_count"] == 0
    kb_ids = [k["id"] for k in data["recent_knowledge_bases"]]
    assert kb_ids == [str(kb2.id), str(kb1.id)]
    assert len(data["recent_knowledge_bases"]) <= 4

    call_ids = [c["id"] for c in data["recent_calls"]]
    assert call_ids == [str(call2.id), str(call1.id)]
    call_map = {c["id"]: c for c in data["recent_calls"]}
    assert call_map[str(call1.id)]["summary"] == "Summary A"
    assert call_map[str(call2.id)]["summary"] is None
    assert len(data["recent_calls"]) <= 4


# ── 2. Empty-tenant case ────────────────────────────────────────────────────────

def test_dashboard_empty_tenant(client, db):
    tenant = _make_tenant(db)

    with _auth_ctx(tenant.id):
        resp = client.get("/api/v1/dashboard")

    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]

    assert data["monthly_calls"] == 0
    assert data["monthly_spent"] == 0.0
    assert data["success_rate_percent"] is None
    assert data["credits"] == pytest.approx(0.0)
    assert data["recent_call_flows"] == []
    assert data["recent_knowledge_bases"] == []
    assert data["recent_calls"] == []


# ── 3. Tenant isolation ─────────────────────────────────────────────────────────

def test_dashboard_tenant_isolation(client, db, any_user_id):
    tenant_a = _make_tenant(db, credits="10.0000")
    tenant_b = _make_tenant(db, credits="99.0000")
    agent_a = _make_agent(db, tenant_a.id)
    agent_b = _make_agent(db, tenant_b.id)
    now = datetime.now(timezone.utc)

    flow_a = _make_call_flow(db, tenant_a.id, agent_a.id, name="A Flow", created_at=now)
    flow_b = _make_call_flow(db, tenant_b.id, agent_b.id, name="B Flow", created_at=now)

    kb_a = _make_kb(db, tenant_a.id, name="A KB", created_at=now)
    kb_b = _make_kb(db, tenant_b.id, name="B KB", created_at=now)

    call_a = _make_call_session(db, tenant_a.id, any_user_id, agent_a.id, start_time=now, cost=5.0)
    call_b = _make_call_session(db, tenant_b.id, any_user_id, agent_b.id, start_time=now, cost=500.0)

    with _auth_ctx(tenant_a.id):
        resp = client.get("/api/v1/dashboard")

    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]

    assert data["monthly_calls"] == 1
    assert data["monthly_spent"] == pytest.approx(5.0)
    assert data["credits"] == pytest.approx(10.0)

    flow_ids = [f["id"] for f in data["recent_call_flows"]]
    kb_ids = [k["id"] for k in data["recent_knowledge_bases"]]
    call_ids = [c["id"] for c in data["recent_calls"]]

    assert str(flow_a.id) in flow_ids
    assert str(flow_b.id) not in flow_ids
    assert str(kb_a.id) in kb_ids
    assert str(kb_b.id) not in kb_ids
    assert str(call_a.id) in call_ids
    assert str(call_b.id) not in call_ids


# ── 3b. limit=4 enforced + most-recent-first ordering ──────────────────────────

def test_dashboard_limit_enforced_most_recent_first(client, db, any_user_id):
    tenant = _make_tenant(db)
    agent = _make_agent(db, tenant.id)
    now = datetime.now(timezone.utc)

    flows = [
        _make_call_flow(db, tenant.id, agent.id, name=f"Flow {i}", created_at=now - timedelta(minutes=(10 - i)))
        for i in range(6)
    ]
    kbs = [
        _make_kb(db, tenant.id, name=f"KB {i}", created_at=now - timedelta(minutes=(10 - i)))
        for i in range(6)
    ]
    calls = [
        _make_call_session(
            db, tenant.id, any_user_id, agent.id,
            start_time=now, created_at=now - timedelta(minutes=(10 - i)), cost=1.0,
        )
        for i in range(6)
    ]

    with _auth_ctx(tenant.id):
        resp = client.get("/api/v1/dashboard")

    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]

    expected_flow_ids = [str(f.id) for f in sorted(flows, key=lambda f: f.created_at, reverse=True)[:4]]
    expected_kb_ids = [str(k.id) for k in sorted(kbs, key=lambda k: k.created_at, reverse=True)[:4]]
    expected_call_ids = [str(c.id) for c in sorted(calls, key=lambda c: c.created_at, reverse=True)[:4]]

    assert len(data["recent_call_flows"]) == 4
    assert [f["id"] for f in data["recent_call_flows"]] == expected_flow_ids

    assert len(data["recent_knowledge_bases"]) == 4
    assert [k["id"] for k in data["recent_knowledge_bases"]] == expected_kb_ids

    assert len(data["recent_calls"]) == 4
    assert [c["id"] for c in data["recent_calls"]] == expected_call_ids


# ── 4. 403 when no tenant selected ──────────────────────────────────────────────

def test_dashboard_no_tenant_selected_returns_403(client, db):
    with _auth_ctx(None):
        resp = client.get("/api/v1/dashboard")

    assert resp.status_code == 403, resp.text
    body = resp.json()
    msg = body.get("detail") or body.get("error", {}).get("message", "")
    assert "No tenant selected" in msg


# ── 5. monthly_calls / monthly_spent are month-scoped ───────────────────────────

def test_dashboard_monthly_metrics_scoped_to_current_month(client, db, any_user_id):
    tenant = _make_tenant(db)
    agent = _make_agent(db, tenant.id)
    now = datetime.now(timezone.utc)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    previous_month_time = month_start - timedelta(days=1)

    _make_call_session(
        db, tenant.id, any_user_id, agent.id, start_time=now, cost=7.0
    )
    _make_call_session(
        db, tenant.id, any_user_id, agent.id, start_time=previous_month_time, cost=1000.0
    )

    with _auth_ctx(tenant.id):
        resp = client.get("/api/v1/dashboard")

    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]

    assert data["monthly_calls"] == 1
    assert data["monthly_spent"] == pytest.approx(7.0)


# ── 6. Response envelope ────────────────────────────────────────────────────────

def test_dashboard_response_envelope(client, db):
    tenant = _make_tenant(db)

    with _auth_ctx(tenant.id):
        resp = client.get("/api/v1/dashboard")

    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["message"] == "Dashboard summary fetched"
    assert body["status_code"] == 200
    assert "data" in body
    assert set(body.keys()) == {"data", "message", "status_code"}
