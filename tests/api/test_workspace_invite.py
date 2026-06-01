"""
Tests for team invitation endpoints.

POST /api/v1/workspace/invite
GET  /api/v1/workspace/invitations
POST /api/v1/accept-invite  (expired token regression)
"""
from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.deps import require_admin
from app.core.security import create_user_token
from app.core.workspace import Workspace
from app.main import app
from app.models.invite import Invite
from app.models.tenant import Tenant
from app.models.user import User


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _workspace_dict(tenant: Tenant) -> dict:
    return {
        "id": str(tenant.id),
        "name": tenant.name,
        "schema_name": tenant.schema_name,
        "status": getattr(tenant, "status", "active") or "active",
        "credits": 10.0,
        "stripe_customer_id": None,
        "stripe_subscription_id": None,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace_tenant(db) -> Tenant:
    t = Tenant(
        name=f"Tenant-{uuid.uuid4().hex[:6]}",
        schema_name=f"s_{uuid.uuid4().hex[:6]}",
        status="active",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@pytest.fixture
def admin_user(db, workspace_tenant) -> User:
    u = User(
        email=f"admin-{uuid.uuid4().hex[:6]}@example.com",
        first_name="Admin",
        last_name="User",
        hashed_password="",
        current_tenant_id=workspace_tenant.id,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


@pytest.fixture
def authed_client(client: TestClient, admin_user: User, workspace_tenant: Tenant):
    """Client with valid JWT + middleware workspace stub + require_admin override."""
    token = create_user_token(
        user_id=admin_user.id,
        email=admin_user.email,
        tenant_id=admin_user.current_tenant_id,
        role="admin",
    )
    workspace = Workspace.from_mapping(_workspace_dict(workspace_tenant))

    async def _fake_load(wid):
        return workspace

    app.dependency_overrides[require_admin] = lambda: admin_user
    with patch("app.middleware.api_key_middleware._load_workspace", side_effect=_fake_load), patch(
        "app.api.api_v1.endpoints.workspace_invites.email_service.send_invite_email",
        return_value=True,
    ):
        client.headers.update({"Authorization": f"Bearer {token}"})
        yield client
    app.dependency_overrides.pop(require_admin, None)
    client.headers.pop("Authorization", None)


# ---------------------------------------------------------------------------
# POST /api/v1/workspace/invite — success
# ---------------------------------------------------------------------------

class TestInviteSuccess:
    def test_invite_creates_pending_record(self, authed_client, db, admin_user):
        email = f"invited-{uuid.uuid4().hex[:6]}@example.com"

        with patch(
            "app.api.api_v1.endpoints.workspace_invites.email_service.send_invite_email",
            return_value=True,
        ) as mock_send:
            resp = authed_client.post("/api/v1/workspace/invite", json={"email": email})

        assert resp.status_code == 201, resp.text
        mock_send.assert_called_once()
        data = resp.json()["data"]
        assert data["email"] == email
        assert data["status"] == "pending"

        invite = db.query(Invite).filter(Invite.email == email).first()
        assert invite is not None
        assert invite.status == "pending"
        assert invite.tenant_id == admin_user.current_tenant_id

        # token must be URL-safe secrets output, not a UUID
        assert len(invite.token) >= 32
        with pytest.raises(ValueError):
            uuid.UUID(invite.token)

        # expires ~7 days from now (SQLite strips tzinfo so compare as naive)
        exp = invite.expires_at
        if exp.tzinfo is not None:
            exp = exp.replace(tzinfo=None)
        delta = exp - datetime.utcnow()
        assert timedelta(days=6, hours=23) < delta < timedelta(days=7, hours=1)

        call_kwargs = mock_send.call_args.kwargs
        assert call_kwargs["email"] == email
        assert call_kwargs["invite_token"] == invite.token
        assert len(call_kwargs["invite_token"]) >= 32

    def test_invite_response_omits_token(self, authed_client):
        email = f"notoken-{uuid.uuid4().hex[:6]}@example.com"
        resp = authed_client.post("/api/v1/workspace/invite", json={"email": email})
        assert resp.status_code == 201
        assert "token" not in resp.json()["data"]


# ---------------------------------------------------------------------------
# POST /api/v1/workspace/invite — duplicate → 409
# ---------------------------------------------------------------------------

class TestDuplicateInvite:
    def test_duplicate_returns_409(self, authed_client):
        email = f"dup-{uuid.uuid4().hex[:6]}@example.com"

        resp1 = authed_client.post("/api/v1/workspace/invite", json={"email": email})
        assert resp1.status_code == 201

        resp2 = authed_client.post("/api/v1/workspace/invite", json={"email": email})
        assert resp2.status_code == 409
        assert "pending" in resp2.json()["error"]["message"].lower()

    def test_invalid_email_returns_422(self, authed_client):
        resp = authed_client.post("/api/v1/workspace/invite", json={"email": "not-an-email"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/workspace/invitations
# ---------------------------------------------------------------------------

class TestListInvitations:
    def test_list_returns_pending_only(self, authed_client, db, admin_user):
        tenant_id = admin_user.current_tenant_id

        pending = Invite(
            email=f"p-{uuid.uuid4().hex[:6]}@example.com",
            tenant_id=tenant_id,
            invited_by=admin_user.id,
            token=str(uuid.uuid4()),
            expires_at=datetime.now(timezone.utc) + timedelta(days=7),
            status="pending",
        )
        accepted = Invite(
            email=f"a-{uuid.uuid4().hex[:6]}@example.com",
            tenant_id=tenant_id,
            invited_by=admin_user.id,
            token=str(uuid.uuid4()),
            expires_at=datetime.now(timezone.utc) + timedelta(days=7),
            status="accepted",
        )
        db.add_all([pending, accepted])
        db.commit()

        resp = authed_client.get("/api/v1/workspace/invitations")
        assert resp.status_code == 200
        items = resp.json()["data"]
        statuses = [i["status"] for i in items]
        assert all(s == "pending" for s in statuses)
        emails = [i["email"] for i in items]
        assert pending.email in emails
        assert accepted.email not in emails

    def test_list_excludes_token_field(self, authed_client, db, admin_user):
        db.add(Invite(
            email=f"sec-{uuid.uuid4().hex[:6]}@example.com",
            tenant_id=admin_user.current_tenant_id,
            invited_by=admin_user.id,
            token=str(uuid.uuid4()),
            expires_at=datetime.now(timezone.utc) + timedelta(days=7),
            status="pending",
        ))
        db.commit()

        resp = authed_client.get("/api/v1/workspace/invitations")
        for item in resp.json()["data"]:
            assert "token" not in item


# ---------------------------------------------------------------------------
# POST /api/v1/accept-invite — expired token regression
# ---------------------------------------------------------------------------

class TestExpiredInviteToken:
    def test_expired_token_returns_error_and_sets_expired_status(self, client, db, admin_user):
        token = secrets.token_urlsafe(32)
        invite = Invite(
            email=f"exp-{uuid.uuid4().hex[:6]}@example.com",
            tenant_id=admin_user.current_tenant_id,
            invited_by=admin_user.id,
            token=token,
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
            status="pending",
        )
        db.add(invite)
        db.commit()

        resp = client.post(f"/api/v1/accept-invite/accept-invite?token={token}&password=secret123")
        assert resp.status_code in (400, 404), resp.text

        db.refresh(invite)
        assert invite.status == "expired"
