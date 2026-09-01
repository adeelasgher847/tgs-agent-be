"""Regression tests: soft-deleted tenants must never reappear in
user-tenant-listing responses.

Bug: `DELETE /api/v1/workspace/{workspace_id}` soft-deletes a tenant
(`Tenant.deleted_at` set) and a second DELETE correctly 404s (proving
`WorkspaceRepository.find_by_id` filters `deleted_at IS NULL`), but
`GET /api/v1/users/my-tenants` and `GET/PUT /api/v1/users/profile` read
`user.tenants` directly — a raw SQLAlchemy relationship with no
`deleted_at` filter — so the deleted tenant still shows up there.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.core.security import create_user_token
from app.models.tenant import Tenant
from app.models.user import User, user_tenant_association


@pytest.fixture
def active_tenant(db) -> Tenant:
    t = Tenant(
        name=f"Active-{uuid.uuid4().hex[:6]}",
        schema_name=f"active_{uuid.uuid4().hex[:6]}",
        status="active",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@pytest.fixture
def deleted_tenant(db) -> Tenant:
    t = Tenant(
        name=f"Deleted-{uuid.uuid4().hex[:6]}",
        schema_name=f"deleted_{uuid.uuid4().hex[:6]}",
        status="active",
        deleted_at=datetime.now(timezone.utc),
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@pytest.fixture
def member_user(db, active_tenant, deleted_tenant) -> User:
    u = User(
        email=f"member-{uuid.uuid4().hex[:6]}@example.com",
        first_name="Member",
        last_name="User",
        hashed_password="",
        current_tenant_id=active_tenant.id,
    )
    db.add(u)
    db.commit()
    db.refresh(u)

    # Associate the user with BOTH tenants (one active, one already
    # soft-deleted) — mirrors "user was a member before the workspace was
    # deleted".
    u.tenants.append(active_tenant)
    u.tenants.append(deleted_tenant)
    db.commit()
    db.refresh(u)
    return u


def _auth_headers(user: User, tenant_id: uuid.UUID) -> dict:
    token = create_user_token(user_id=user.id, email=user.email, tenant_id=tenant_id)
    return {"Authorization": f"Bearer {token}"}


class TestMyTenantsExcludesSoftDeleted:
    def test_my_tenants_excludes_soft_deleted_tenant(
        self, client: TestClient, member_user, active_tenant, deleted_tenant
    ):
        resp = client.get(
            "/api/v1/users/my-tenants",
            headers=_auth_headers(member_user, active_tenant.id),
        )
        assert resp.status_code == 200, resp.text
        tenant_ids = {t["id"] for t in resp.json()["data"]["tenants"]}
        assert str(active_tenant.id) in tenant_ids
        assert (
            str(deleted_tenant.id) not in tenant_ids
        ), "Soft-deleted tenant must not appear in GET /users/my-tenants"


class TestProfileExcludesSoftDeleted:
    def test_get_profile_excludes_soft_deleted_tenant(
        self, client: TestClient, member_user, active_tenant, deleted_tenant
    ):
        resp = client.get(
            "/api/v1/users/profile",
            headers=_auth_headers(member_user, active_tenant.id),
        )
        assert resp.status_code == 200, resp.text
        tenant_ids = {t["id"] for t in resp.json()["data"]["tenants"]}
        assert str(active_tenant.id) in tenant_ids
        assert (
            str(deleted_tenant.id) not in tenant_ids
        ), "Soft-deleted tenant must not appear in GET /users/profile"

    def test_put_profile_excludes_soft_deleted_tenant(
        self, client: TestClient, member_user, active_tenant, deleted_tenant
    ):
        resp = client.put(
            "/api/v1/users/profile",
            json={"first_name": "Updated"},
            headers=_auth_headers(member_user, active_tenant.id),
        )
        assert resp.status_code == 200, resp.text
        tenant_ids = {t["id"] for t in resp.json()["data"]["tenants"]}
        assert str(active_tenant.id) in tenant_ids
        assert (
            str(deleted_tenant.id) not in tenant_ids
        ), "Soft-deleted tenant must not appear in PUT /users/profile response"


class TestTenantMemberRemovalEndpointAbsent:
    def test_delete_member_route_is_not_exposed(self, client: TestClient):
        member_id = "123e4567-e89b-12d3-a456-426614174000"
        resp = client.delete(f"/api/v1/users/tenant-members/{member_id}")
        assert resp.status_code == 404, resp.text

        schema = client.app.openapi()
        path = schema["paths"].get("/api/v1/users/tenant-members/{member_id}")
        assert path is None


class TestSoftDeleteTenantUnderUser:
    def test_user_can_soft_delete_own_tenant_by_id(
        self, client: TestClient, db, member_user, active_tenant
    ):
        headers = _auth_headers(member_user, active_tenant.id)

        # Ensure active_tenant is present in my-tenants first
        resp = client.get("/api/v1/users/my-tenants", headers=headers)
        assert resp.status_code == 200
        tenant_ids = {t["id"] for t in resp.json()["data"]["tenants"]}
        assert str(active_tenant.id) in tenant_ids

        # Soft delete tenant by ID under /users/tenants/{tenant_id}
        resp = client.delete(
            f"/api/v1/users/tenants/{active_tenant.id}",
            headers=headers,
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["tenant_id"] == str(active_tenant.id)
        assert data["status"] == "deleted"

        # Verify DB state
        db.refresh(active_tenant)
        assert active_tenant.deleted_at is not None

        # Verify user's current_tenant_id was reset
        db.refresh(member_user)
        assert member_user.current_tenant_id is None

        # Verify it no longer shows in my-tenants
        resp = client.get("/api/v1/users/my-tenants", headers=headers)
        assert resp.status_code == 200
        tenant_ids = {t["id"] for t in resp.json()["data"]["tenants"]}
        assert str(active_tenant.id) not in tenant_ids

        # Attempting to delete again returns 404
        resp = client.delete(
            f"/api/v1/users/tenants/{active_tenant.id}",
            headers=headers,
        )
        assert resp.status_code == 404

    def test_user_cannot_delete_other_user_tenant(
        self, client: TestClient, db, member_user
    ):
        # Create another tenant not belonging to member_user
        other_tenant = Tenant(
            name=f"Other-{uuid.uuid4().hex[:6]}",
            schema_name=f"other_{uuid.uuid4().hex[:6]}",
            status="active",
        )
        db.add(other_tenant)
        db.commit()
        db.refresh(other_tenant)

        # member_user attempts to delete other_tenant
        headers = _auth_headers(member_user, uuid.uuid4())
        resp = client.delete(
            f"/api/v1/users/tenants/{other_tenant.id}",
            headers=headers,
        )
        assert resp.status_code == 403, resp.text

    def test_delete_nonexistent_tenant_returns_404(
        self, client: TestClient, member_user
    ):
        headers = _auth_headers(member_user, uuid.uuid4())
        resp = client.delete(
            f"/api/v1/users/tenants/{uuid.uuid4()}",
            headers=headers,
        )
        assert resp.status_code == 404

    def test_soft_delete_via_alias_endpoint(
        self, client: TestClient, db, member_user
    ):
        new_tenant = Tenant(
            name=f"AliasRoute-{uuid.uuid4().hex[:6]}",
            schema_name=f"ar_{uuid.uuid4().hex[:6]}",
            status="active",
        )
        db.add(new_tenant)
        db.commit()
        db.refresh(new_tenant)
        member_user.tenants.append(new_tenant)
        db.commit()

        headers = _auth_headers(member_user, new_tenant.id)
        resp = client.delete(
            f"/api/v1/users/my-tenants/{new_tenant.id}",
            headers=headers,
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["status"] == "deleted"

        db.refresh(new_tenant)
        assert new_tenant.deleted_at is not None

    def test_soft_delete_resets_other_members_current_tenant(
        self, client: TestClient, db, member_user, active_tenant
    ):
        other_user = User(
            email=f"coworker-{uuid.uuid4().hex[:6]}@example.com",
            first_name="Co",
            last_name="Worker",
            hashed_password="",
            current_tenant_id=active_tenant.id,
        )
        db.add(other_user)
        db.commit()
        db.refresh(other_user)

        headers = _auth_headers(member_user, active_tenant.id)
        resp = client.delete(
            f"/api/v1/users/tenants/{active_tenant.id}",
            headers=headers,
        )
        assert resp.status_code == 200

        db.refresh(other_user)
        assert other_user.current_tenant_id is None

    def test_openapi_schema_has_delete_tenant_endpoints(self, client: TestClient):
        schema = client.app.openapi()
        user_delete_path = schema["paths"].get("/api/v1/users/tenants/{tenant_id}")
        assert user_delete_path is not None
        assert "delete" in user_delete_path


