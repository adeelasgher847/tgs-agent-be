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
from app.models.user import User


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
