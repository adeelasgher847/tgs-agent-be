#!/usr/bin/env python3
"""Idempotent dev-workspace seed script.

Creates one test tenant (workspace) + one admin user linked with the 'admin'
role in user_tenant_association. Safe to run multiple times — existing rows
are left untouched.

Usage:
    python scripts/seed_dev_workspace.py

Requirements:
    - DATABASE_URL env var or .env file readable by app.core.config.settings
    - Migrations already applied: alembic upgrade head
    - Roles seeded by this script itself (no separate step needed)

Verification (after running):
    The seed user should pass require_admin() on any admin-gated route:
        POST /api/v1/users/login
        body: {"email": "admin@example.com", "password": "dev-password-change-me"}
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

# Allow importing from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.role import Role
from app.models.tenant import Tenant
from app.models.user import User, user_tenant_association
from app.core.security import get_password_hash

# ------------------------------------------------------------------ constants

SEED_TENANT_NAME = "dev-workspace"
# Use a domain accepted by Pydantic EmailStr (reserved TLDs like .local fail on /users/profile).
SEED_ADMIN_EMAIL = "admin@example.com"
LEGACY_SEED_EMAIL = "admin@dev.local"
SEED_ADMIN_PASSWORD = "dev-password-change-me"
SEED_ADMIN_FIRST = "Dev"
SEED_ADMIN_LAST = "Admin"

# Canonical role names — must match values in the role table
_ALL_ROLES = [
    ("owner",    "Owner role with full access to tenant"),
    ("admin",    "Administrator role with full access"),
    ("member",   "Regular member role with limited access"),
    ("config",   "Configure workspace settings; cannot manage users"),
    ("readonly", "Read-only access; blocked from mutating endpoints"),
]


def _ensure_roles(db: Session) -> dict[str, Role]:
    """Insert any missing roles and return a name→Role mapping."""
    roles: dict[str, Role] = {}
    for name, description in _ALL_ROLES:
        role = db.execute(select(Role).where(Role.name == name)).scalar_one_or_none()
        if role is None:
            role = Role(name=name, description=description)
            db.add(role)
            db.flush()
            print(f"[seed] Created role '{name}'")
        else:
            print(f"[seed] Role '{name}' already exists")
        roles[name] = role
    return roles


def seed(db: Session) -> None:
    # Step 1 — ensure all canonical roles exist
    roles = _ensure_roles(db)

    # Step 2 — tenant
    tenant = db.execute(
        select(Tenant).where(
            Tenant.name == SEED_TENANT_NAME,
            Tenant.deleted_at.is_(None),
        )
    ).scalar_one_or_none()

    if tenant is None:
        tenant = Tenant(
            id=uuid.uuid4(),
            name=SEED_TENANT_NAME,
            schema_name="dev_workspace_schema",
            status="active",
            credits=0,
        )
        db.add(tenant)
        db.flush()
        print(f"[seed] Created tenant '{SEED_TENANT_NAME}' id={tenant.id}")
    else:
        print(f"[seed] Tenant '{SEED_TENANT_NAME}' already exists id={tenant.id}")

    # Step 3 — user (bcrypt hash so the account can log in via the API)
    user = db.execute(
        select(User).where(User.email == SEED_ADMIN_EMAIL)
    ).scalar_one_or_none()

    if user is None:
        legacy = db.execute(
            select(User).where(User.email == LEGACY_SEED_EMAIL)
        ).scalar_one_or_none()
        if legacy is not None:
            legacy.email = SEED_ADMIN_EMAIL
            db.flush()
            user = legacy
            print(
                f"[seed] Migrated legacy email '{LEGACY_SEED_EMAIL}' -> '{SEED_ADMIN_EMAIL}'"
            )

    if user is None:
        user = User(
            id=uuid.uuid4(),
            first_name=SEED_ADMIN_FIRST,
            last_name=SEED_ADMIN_LAST,
            email=SEED_ADMIN_EMAIL,
            hashed_password=get_password_hash(SEED_ADMIN_PASSWORD),
            current_tenant_id=tenant.id,
        )
        db.add(user)
        db.flush()
        print(f"[seed] Created admin user '{SEED_ADMIN_EMAIL}' id={user.id}")
    else:
        print(f"[seed] Admin user '{SEED_ADMIN_EMAIL}' already exists id={user.id}")

    # Step 4 — user_tenant_association with admin role_id
    admin_role = roles["admin"]
    existing_link = db.execute(
        select(user_tenant_association).where(
            user_tenant_association.c.user_id == user.id,
            user_tenant_association.c.tenant_id == tenant.id,
        )
    ).first()

    if existing_link is None:
        db.execute(
            user_tenant_association.insert().values(
                user_id=user.id,
                tenant_id=tenant.id,
                is_creator=True,
                role_id=admin_role.id,
            )
        )
        print(f"[seed] Linked admin user to '{SEED_TENANT_NAME}' with role 'admin'")
    else:
        # Ensure role_id is set to admin even if the row existed without it
        if existing_link.role_id != admin_role.id:
            db.execute(
                user_tenant_association.update()
                .where(
                    user_tenant_association.c.user_id == user.id,
                    user_tenant_association.c.tenant_id == tenant.id,
                )
                .values(role_id=admin_role.id, is_creator=True)
            )
            print("[seed] Updated existing user-tenant link to role 'admin'")
        else:
            print("[seed] User-tenant link already exists with role 'admin'")

    db.commit()
    print("[seed] Done.")
    print(f"\n[seed] Login credentials:")
    print(f"       email:    {SEED_ADMIN_EMAIL}")
    print(f"       password: {SEED_ADMIN_PASSWORD}")


if __name__ == "__main__":
    db: Session = SessionLocal()
    try:
        seed(db)
    except Exception as exc:
        db.rollback()
        print(f"[seed] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        db.close()
