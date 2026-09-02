"""Tests for PUT /api/v2/workspace/members/{user_id}/role.

Coverage:
  - admin can promote/demote another member -> 200, role_id updated, cache invalidated
  - self-demotion below current rank -> 400 (admin can't drop their own role)
  - self-reassignment to admin (no-op rank-wise) -> 200
  - invalid role name -> 400
  - target not a member of the workspace -> 404
  - audit event fired with old/new value
"""
from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.testclient import TestClient

from app.api.deps import get_db, require_admin, require_readonly, require_tenant
from app.core.exception_handlers import register_exception_handlers
from app.models.role import Role
from app.models.tenant import Tenant
from app.models.user import User, user_tenant_association
from app.services import role_service


@pytest.fixture(autouse=True)
def seed_canonical_roles(db):
    """In production the migration guarantees all 5 canonical roles exist;
    this throwaway test DB only has whatever Base.metadata.create_all() built,
    so seed the catalog the same way the role-update endpoint expects it."""
    for name in role_service.CANONICAL_ROLES:
        if db.query(Role).filter(Role.name == name).first() is None:
            db.add(Role(name=name, description=name))
    db.commit()


@pytest.fixture
def tenant(db) -> Tenant:
    t = Tenant(
        name=f"member-role-{uuid.uuid4().hex[:8]}",
        schema_name=f"s_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


def _get_or_create_role(db, name: str) -> Role:
    role = db.query(Role).filter(Role.name == name).first()
    if role is None:
        role = Role(name=name, description=name)
        db.add(role)
        db.commit()
        db.refresh(role)
    return role


def _make_member(db, tenant_id, role_name: str, *, is_creator: bool = False) -> User:
    role = _get_or_create_role(db, role_name)
    user = User(
        email=f"{role_name}-{uuid.uuid4().hex[:8]}@example.com",
        first_name="Member",
        last_name="Test",
        hashed_password="x",
        current_tenant_id=tenant_id,
    )
    db.add(user)
    db.flush()
    db.execute(
        user_tenant_association.insert().values(
            user_id=user.id, tenant_id=tenant_id, role_id=role.id, is_creator=is_creator,
        )
    )
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def admin_user(db, tenant) -> User:
    return _make_member(db, tenant.id, "admin", is_creator=True)


def _client(db, admin_user, admin_override=None) -> TestClient:
    from app.api.v2.routers.workspace import v2_router

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(v2_router, prefix="/workspace")
    mini.dependency_overrides[require_admin] = admin_override or (lambda: admin_user)
    mini.dependency_overrides[get_db] = lambda: db
    return TestClient(mini, raise_server_exceptions=False)


def _role_id_for(db, user_id, tenant_id):
    row = db.execute(
        user_tenant_association.select().where(
            user_tenant_association.c.user_id == user_id,
            user_tenant_association.c.tenant_id == tenant_id,
        )
    ).first()
    return row.role_id if row is not None else None


class TestUpdateMemberRole:
    def test_admin_promotes_member_to_manager(self, db, tenant, admin_user):
        target = _make_member(db, tenant.id, "config_only")
        client = _client(db, admin_user)

        with patch("app.api.v2.routers.workspace.rbac_cache_service.invalidate") as mock_invalidate:
            resp = client.put(
                f"/workspace/members/{target.id}/role", json={"role": "manager"}
            )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["role"] == "manager"
        assert body["user_id"] == str(target.id)

        manager_role = _get_or_create_role(db, "manager")
        assert _role_id_for(db, target.id, tenant.id) == manager_role.id
        mock_invalidate.assert_called_once_with(target.id, tenant.id)

    def test_admin_demotes_other_member_to_read_only(self, db, tenant, admin_user):
        target = _make_member(db, tenant.id, "manager")
        client = _client(db, admin_user)

        resp = client.put(
            f"/workspace/members/{target.id}/role", json={"role": "read_only"}
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["role"] == "read_only"

    def test_self_demotion_below_current_rank_returns_400(self, db, tenant, admin_user):
        client = _client(db, admin_user)
        resp = client.put(
            f"/workspace/members/{admin_user.id}/role", json={"role": "config_only"}
        )
        assert resp.status_code == 400, resp.text
        assert "self-demote" in resp.json()["error"]["message"].lower()

    def test_self_reassignment_to_same_rank_allowed(self, db, tenant, admin_user):
        client = _client(db, admin_user)
        resp = client.put(
            f"/workspace/members/{admin_user.id}/role", json={"role": "admin"}
        )
        assert resp.status_code == 200, resp.text

    def test_invalid_role_name_returns_400(self, db, tenant, admin_user):
        target = _make_member(db, tenant.id, "config_only")
        client = _client(db, admin_user)
        resp = client.put(
            f"/workspace/members/{target.id}/role", json={"role": "superadmin"}
        )
        assert resp.status_code == 400, resp.text

    def test_target_not_a_member_returns_404(self, db, tenant, admin_user):
        non_member_id = uuid.uuid4()
        client = _client(db, admin_user)
        resp = client.put(
            f"/workspace/members/{non_member_id}/role", json={"role": "manager"}
        )
        assert resp.status_code == 404, resp.text

    def test_role_update_fires_audit_event(self, db, tenant, admin_user):
        target = _make_member(db, tenant.id, "config_only")
        client = _client(db, admin_user)

        with patch("app.api.v2.routers.workspace.log_audit_event") as mock_audit:
            resp = client.put(
                f"/workspace/members/{target.id}/role", json={"role": "read_only"}
            )

        assert resp.status_code == 200, resp.text
        mock_audit.assert_called_once()
        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["action"] == "workspace.member_role_updated"
        assert call_kwargs["resource_id"] == target.id
        assert call_kwargs["new_value"] == {"role": "read_only"}


class TestRemoveMember:
    def test_admin_removes_member_from_workspace(self, db, tenant, admin_user):
        target = _make_member(db, tenant.id, "read_only")
        client = _client(db, admin_user)

        with patch(
            "app.api.v2.routers.workspace.rbac_cache_service.invalidate"
        ) as mock_invalidate:
            resp = client.delete(f"/workspace/members/{target.id}")

        assert resp.status_code == 200, resp.text
        mock_invalidate.assert_called_once_with(target.id, tenant.id)
        assert resp.json()["data"]["user_id"] == str(target.id)
        assert db.query(User).filter(User.id == target.id).first() is not None
        membership = db.execute(
            user_tenant_association.select().where(
                user_tenant_association.c.user_id == target.id,
                user_tenant_association.c.tenant_id == tenant.id,
            )
        ).first()
        assert membership is not None
        assert membership.removed_at is not None

        second_resp = client.delete(f"/workspace/members/{target.id}")
        assert second_resp.status_code == 404

    def test_workspace_creator_cannot_be_removed(self, db, tenant, admin_user):
        actor = _make_member(db, tenant.id, "admin")
        client = _client(db, actor)

        resp = client.delete(f"/workspace/members/{admin_user.id}")

        assert resp.status_code == 400, resp.text
        membership = db.execute(
            user_tenant_association.select().where(
                user_tenant_association.c.user_id == admin_user.id,
                user_tenant_association.c.tenant_id == tenant.id,
            )
        ).first()
        assert membership is not None
        assert membership.removed_at is None

    def test_admin_cannot_remove_themselves(self, db, tenant, admin_user):
        client = _client(db, admin_user)

        resp = client.delete(f"/workspace/members/{admin_user.id}")

        assert resp.status_code == 400, resp.text
        membership = db.execute(
            user_tenant_association.select().where(
                user_tenant_association.c.user_id == admin_user.id,
                user_tenant_association.c.tenant_id == tenant.id,
            )
        ).first()
        assert membership is not None
        assert membership.removed_at is None

    def test_removal_invalidates_real_rbac_cache(self, db, tenant, admin_user):
        target = _make_member(db, tenant.id, "read_only")
        from app.api.v2.routers.workspace import v2_router

        client_app = FastAPI()
        register_exception_handlers(client_app)
        client_app.include_router(v2_router, prefix="/workspace")
        client_app.dependency_overrides[require_admin] = lambda: admin_user
        client_app.dependency_overrides[require_tenant] = lambda: target
        client_app.dependency_overrides[get_db] = lambda: db

        @client_app.get("/protected")
        def protected_endpoint(user=Depends(require_readonly)):
            return {"user_id": str(user.id)}

        client = TestClient(client_app, raise_server_exceptions=False)

        cached_response = client.get("/protected")
        assert cached_response.status_code == 200, cached_response.text

        remove_response = client.delete(f"/workspace/members/{target.id}")
        assert remove_response.status_code == 200, remove_response.text

        denied_response = client.get("/protected")
        assert denied_response.status_code == 403, denied_response.text

    def test_unauthorized_member_removal_returns_403(self, db, tenant, admin_user):
        target = _make_member(db, tenant.id, "read_only")
        client = _client(
            db,
            admin_user,
            admin_override=lambda: (_ for _ in ()).throw(
                HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
            ),
        )

        resp = client.delete(f"/workspace/members/{target.id}")

        assert resp.status_code == 403

    def test_cross_tenant_member_removal_returns_404(self, db, tenant, admin_user):
        other_tenant = Tenant(
            name=f"other-member-{uuid.uuid4().hex[:8]}",
            schema_name=f"s_{uuid.uuid4().hex[:8]}",
            status="active",
        )
        db.add(other_tenant)
        db.commit()
        target = _make_member(db, other_tenant.id, "read_only")
        client = _client(db, admin_user)

        resp = client.delete(f"/workspace/members/{target.id}")

        assert resp.status_code == 404
        assert _role_id_for(db, target.id, other_tenant.id) is not None

    def test_nonexistent_member_removal_returns_404(self, db, tenant, admin_user):
        client = _client(db, admin_user)

        resp = client.delete(f"/workspace/members/{uuid.uuid4()}")

        assert resp.status_code == 404
