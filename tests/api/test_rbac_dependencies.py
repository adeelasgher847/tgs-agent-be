"""RBAC dependency matrix — every canonical role against every require_* dependency.

Per the RBAC hardening ticket: 5 roles x 5 dependency functions = 25 minimum
cases (test_role_dependency_matrix below). Plus coverage for the
non-tabular behaviors: workspace-owner override, default-to-read_only when
no role is assigned, and rejection of non-members.
"""
from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException

from app.api.deps import (
    require_admin,
    require_manager,
    require_config,
    require_readonly,
    require_billing,
)
from app.models.role import Role
from app.models.tenant import Tenant
from app.models.user import User, user_tenant_association
from app.services import role_service

DEPENDENCIES = {
    "admin": require_admin,
    "manager": require_manager,
    "config_only": require_config,
    "read_only": require_readonly,
    "billing_only": require_billing,
}

# EXPECTED[caller_role][dependency_required] -> should the call succeed?
# admin > manager > config_only > read_only is a strict chain; billing_only
# sits outside it, satisfied only by itself, admin, and manager.
EXPECTED = {
    "admin": {
        "admin": True, "manager": True, "config_only": True,
        "read_only": True, "billing_only": True,
    },
    "manager": {
        "admin": False, "manager": True, "config_only": True,
        "read_only": True, "billing_only": True,
    },
    "config_only": {
        "admin": False, "manager": False, "config_only": True,
        "read_only": True, "billing_only": False,
    },
    "read_only": {
        "admin": False, "manager": False, "config_only": False,
        "read_only": True, "billing_only": False,
    },
    "billing_only": {
        "admin": False, "manager": False, "config_only": False,
        "read_only": False, "billing_only": True,
    },
}


@pytest.fixture
def tenant(db) -> Tenant:
    t = Tenant(
        name=f"rbac-{uuid.uuid4().hex[:8]}",
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


def _make_member(db, tenant_id, role_name: str | None, *, is_creator: bool = False) -> User:
    """Create a user with a user_tenant_association row for ``tenant_id``.

    role_name=None creates the row with role_id left NULL (the "no role
    assigned yet" case, distinct from "not a member at all" — no row).
    """
    role_id = _get_or_create_role(db, role_name).id if role_name else None

    user = User(
        email=f"{role_name or 'norole'}-{uuid.uuid4().hex[:8]}@example.com",
        first_name="RBAC",
        last_name="Test",
        hashed_password="x",
        current_tenant_id=tenant_id,
    )
    db.add(user)
    db.flush()
    db.execute(
        user_tenant_association.insert().values(
            user_id=user.id, tenant_id=tenant_id, role_id=role_id, is_creator=is_creator,
        )
    )
    db.commit()
    db.refresh(user)
    return user


# ─────────────────────────────────────────────────── the 25-case matrix ──

@pytest.mark.parametrize("dependency_name", sorted(DEPENDENCIES))
@pytest.mark.parametrize("caller_role", sorted(EXPECTED))
def test_role_dependency_matrix(db, tenant, caller_role, dependency_name):
    user = _make_member(db, tenant.id, caller_role)
    dependency = DEPENDENCIES[dependency_name]
    should_pass = EXPECTED[caller_role][dependency_name]

    if should_pass:
        result = dependency(user=user, db=db)
        assert result is user
    else:
        with pytest.raises(HTTPException) as exc_info:
            dependency(user=user, db=db)
        assert exc_info.value.status_code == 403
        detail = exc_info.value.detail
        assert detail["code"] == "forbidden"
        assert detail["role_required"] == dependency_name
        assert detail["user_role"] == caller_role
        assert "message" in detail


# ───────────────────────────────────────────── owner override (is_creator) ──

class TestWorkspaceOwnerOverride:
    def test_creator_passes_require_admin_regardless_of_stored_role(self, db, tenant):
        creator = _make_member(db, tenant.id, "read_only", is_creator=True)
        result = require_admin(user=creator, db=db)
        assert result is creator

    def test_creator_with_no_role_row_at_all_role_id_still_passes_admin(self, db, tenant):
        creator = _make_member(db, tenant.id, None, is_creator=True)
        result = require_admin(user=creator, db=db)
        assert result is creator

    def test_non_creator_with_admin_named_role_is_unaffected(self, db, tenant):
        # Sanity check: the override is additive, not a replacement of normal
        # role-based admin access.
        admin_user = _make_member(db, tenant.id, "admin", is_creator=False)
        result = require_admin(user=admin_user, db=db)
        assert result is admin_user


# ────────────────────────────────────── default-to-read_only, never rejected ──

class TestDefaultRole:
    def test_member_with_no_role_assigned_defaults_to_read_only(self, db, tenant):
        member = _make_member(db, tenant.id, None)
        result = require_readonly(user=member, db=db)
        assert result is member

    def test_member_with_no_role_assigned_fails_config(self, db, tenant):
        member = _make_member(db, tenant.id, None)
        with pytest.raises(HTTPException) as exc_info:
            require_config(user=member, db=db)
        assert exc_info.value.status_code == 403
        assert exc_info.value.detail["user_role"] == "read_only"


# ───────────────────────────────────────────────────── non-member rejection ──

class TestNonMemberRejection:
    @pytest.mark.parametrize("dependency_name", sorted(DEPENDENCIES))
    def test_user_with_no_membership_row_rejected(self, db, tenant, dependency_name):
        stranger = User(
            email=f"stranger-{uuid.uuid4().hex[:8]}@example.com",
            first_name="No",
            last_name="Membership",
            hashed_password="x",
            current_tenant_id=tenant.id,
        )
        db.add(stranger)
        db.commit()
        db.refresh(stranger)

        dependency = DEPENDENCIES[dependency_name]
        with pytest.raises(HTTPException) as exc_info:
            dependency(user=stranger, db=db)
        assert exc_info.value.status_code == 403
        assert "not a member" in str(exc_info.value.detail).lower()
