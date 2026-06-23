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

from app.api.deps import require_admin_or_api_key
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

    app.dependency_overrides[require_admin_or_api_key] = lambda: admin_user
    with patch("app.middleware.api_key_middleware._load_workspace", side_effect=_fake_load), patch(
        "app.api.api_v1.endpoints.workspace_invites.email_service.send_invite_email",
        return_value=True,
    ):
        client.headers.update({"Authorization": f"Bearer {token}"})
        yield client
    app.dependency_overrides.pop(require_admin_or_api_key, None)
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

    def test_invalid_email_returns_400(self, authed_client):
        resp = authed_client.post("/api/v1/workspace/invite", json={"email": "not-an-email"})
        assert resp.status_code == 400


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


# ---------------------------------------------------------------------------
# API Key authentication for invite endpoints
# ---------------------------------------------------------------------------

class TestInviteWithApiKey:
    def test_invite_with_api_key_resolves_creator(self, client, db, admin_user, workspace_tenant):
        from app.models.user import user_tenant_association
        # Link admin_user as creator in db
        db.execute(
            user_tenant_association.insert().values(
                user_id=admin_user.id,
                tenant_id=workspace_tenant.id,
                is_creator=True,
            )
        )
        db.commit()

        api_key_payload = {
            "api_key_id": str(uuid.uuid4()),
            "tenant_id": str(workspace_tenant.id),
            "key_is_active": True,
            "workspace": {
                "id": str(workspace_tenant.id),
                "name": workspace_tenant.name,
                "schema_name": workspace_tenant.schema_name,
                "status": "active",
                "credits": 10.0,
                "stripe_customer_id": None,
                "stripe_subscription_id": None,
            },
        }

        async def _resolve(_key_hash, _workspace_id):
            return api_key_payload

        email = f"api-invited-{uuid.uuid4().hex[:6]}@example.com"
        headers = {"x-api-key": "testkey", "x-workspace-id": str(workspace_tenant.id)}

        with patch(
            "app.middleware.api_key_middleware._resolve_api_key",
            side_effect=_resolve,
        ), patch(
            "app.api.api_v1.endpoints.workspace_invites.email_service.send_invite_email",
            return_value=True,
        ):
            resp = client.post(
                "/api/v1/workspace/invite",
                json={"email": email},
                headers=headers,
            )
            assert resp.status_code == 201, resp.text
            assert resp.json()["data"]["email"] == email

            # Check GET works too
            resp_get = client.get("/api/v1/workspace/invitations", headers=headers)
            assert resp_get.status_code == 200, resp_get.text
            emails = [item["email"] for item in resp_get.json()["data"]]
            assert email in emails


# ---------------------------------------------------------------------------
# Test Specifying Role during Invite
# ---------------------------------------------------------------------------

class TestRoleBasedInvitations:
    def test_invite_with_role_id_success(self, authed_client, db, admin_user):
        from app.models.role import Role
        # Get manager role ID
        manager_role = db.query(Role).filter(Role.name == "manager").first()
        assert manager_role is not None

        email = f"invited-manager-{uuid.uuid4().hex[:6]}@example.com"
        resp = authed_client.post(
            "/api/v1/workspace/invite",
            json={"email": email, "role_id": str(manager_role.id)}
        )
        assert resp.status_code == 201, resp.text
        data = resp.json()["data"]
        assert data["role_id"] == str(manager_role.id)

        # Check in DB
        invite = db.query(Invite).filter(Invite.email == email).first()
        assert invite is not None
        assert invite.role_id == manager_role.id

    def test_invite_with_invalid_role_id_returns_404(self, authed_client):
        email = f"invited-invalid-{uuid.uuid4().hex[:6]}@example.com"
        resp = authed_client.post(
            "/api/v1/workspace/invite",
            json={"email": email, "role_id": str(uuid.uuid4())}
        )
        assert resp.status_code == 404, resp.text

    def test_accept_invite_assigns_correct_role(self, client, db, admin_user):
        from app.models.role import Role
        from app.models.user import user_tenant_association

        manager_role = db.query(Role).filter(Role.name == "manager").first()
        assert manager_role is not None

        token = secrets.token_urlsafe(32)
        email = f"accept-role-{uuid.uuid4().hex[:6]}@example.com"
        invite = Invite(
            email=email,
            tenant_id=admin_user.current_tenant_id,
            invited_by=admin_user.id,
            role_id=manager_role.id,
            token=token,
            expires_at=datetime.now(timezone.utc) + timedelta(days=7),
            status="pending",
        )
        db.add(invite)
        db.commit()

        resp = client.post(f"/api/v1/accept-invite/accept-invite?token={token}&password=secret123")
        assert resp.status_code == 200, resp.text


        # Verify that the accepted user has the manager role in this tenant
        accepted_user = db.query(User).filter(User.email == email).first()
        assert accepted_user is not None

        assoc = db.query(user_tenant_association).filter(
            user_tenant_association.c.user_id == accepted_user.id,
            user_tenant_association.c.tenant_id == admin_user.current_tenant_id
        ).first()
        assert assoc is not None
        assert assoc.role_id == manager_role.id


