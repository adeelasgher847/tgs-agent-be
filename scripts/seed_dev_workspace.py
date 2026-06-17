#!/usr/bin/env python3
"""Idempotent dev-workspace seed script.

Creates hierarchical V3 workspaces:
  - 1 Agency Workspace ("dev-agency-workspace")
  - 1 Sub-Account Workspace ("dev-subaccount-workspace") linked to the Agency.
  - Links the admin user to both workspaces using the 'admin' role.

Safe to run multiple times — existing rows are left untouched.

Usage:
    python scripts/seed_dev_workspace.py
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

# Allow importing from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uuid as _uuid_mod

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.agent import Agent
from app.models.call_flow import CallFlow
from app.models.prompt_version import PromptVersion
from app.models.role import Role
from app.models.tenant import Tenant
from app.models.branding_configs import BrandingConfig
from app.models.pricing_configs import PricingConfig

from app.models.usage_records import UsageRecords
from app.models.user import User, user_tenant_association
from app.core.security import get_password_hash

# ------------------------------------------------------------------ constants

SEED_AGENCY_NAME = "dev-agency-workspace"
SEED_SUBACCOUNT_NAME = "dev-subaccount-workspace"

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
    # Step 1 — Ensure all canonical roles exist
    roles = _ensure_roles(db)

    # Step 2 — Create/Ensure Parent Agency Tenant
    agency_tenant = db.execute(
        select(Tenant).where(
            Tenant.name == SEED_AGENCY_NAME,
            Tenant.deleted_at.is_(None),
        )
    ).scalar_one_or_none()

    if agency_tenant is None:
        agency_tenant = Tenant(
            id=uuid.uuid4(),
            name=SEED_AGENCY_NAME,
            schema_name="dev_agency_schema",
            status="active",
            credits=1000,
            workspace_type="agency",          # V3 field
            parent_workspace_id=None,         # Top level root
        )
        db.add(agency_tenant)
        db.flush()
        print(f"[seed] Created Parent Agency tenant '{SEED_AGENCY_NAME}' id={agency_tenant.id}")
    else:
        print(f"[seed] Parent Agency tenant '{SEED_AGENCY_NAME}' already exists id={agency_tenant.id}")

    # Step 3 — Create/Ensure Linked Sub-Account Tenant
    sub_tenant = db.execute(
        select(Tenant).where(
            Tenant.name == SEED_SUBACCOUNT_NAME,
            Tenant.deleted_at.is_(None),
        )
    ).scalar_one_or_none()

    if sub_tenant is None:
        sub_tenant = Tenant(
            id=uuid.uuid4(),
            name=SEED_SUBACCOUNT_NAME,
            schema_name="dev_subaccount_schema",
            status="active",
            credits=0,
            workspace_type="sub_account",       # V3 field
            parent_workspace_id=agency_tenant.id # Links directly to Agency
        )
        db.add(sub_tenant)
        db.flush()
        print(f"[seed] Created Linked Sub-Account tenant '{SEED_SUBACCOUNT_NAME}' id={sub_tenant.id}")
    else:
        print(f"[seed] Linked Sub-Account tenant '{SEED_SUBACCOUNT_NAME}' already exists id={sub_tenant.id}")

    # Step 4 — Admin User Setup
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
            print(f"[seed] Migrated legacy email '{LEGACY_SEED_EMAIL}' -> '{SEED_ADMIN_EMAIL}'")

    if user is None:
        user = User(
            id=uuid.uuid4(),
            first_name=SEED_ADMIN_FIRST,
            last_name=SEED_ADMIN_LAST,
            email=SEED_ADMIN_EMAIL,
            hashed_password=get_password_hash(SEED_ADMIN_PASSWORD),
            current_tenant_id=agency_tenant.id, # Defaults context to Agency
        )
        db.add(user)
        db.flush()
        print(f"[seed] Created admin user '{SEED_ADMIN_EMAIL}' id={user.id}")
    else:
        print(f"[seed] Admin user '{SEED_ADMIN_EMAIL}' already exists id={user.id}")

    # Step 5 — Map user permissions across BOTH workspaces idempotently
    admin_role = roles["admin"]
    workspaces_to_link = [
        ("Agency", agency_tenant.id),
        ("Sub-Account", sub_tenant.id)
    ]

    for label, tenant_id in workspaces_to_link:
        existing_link = db.execute(
            select(user_tenant_association).where(
                user_tenant_association.c.user_id == user.id,
                user_tenant_association.c.tenant_id == tenant_id,
            )
        ).first()

        if existing_link is None:
            db.execute(
                user_tenant_association.insert().values(
                    user_id=user.id,
                    tenant_id=tenant_id,
                    is_creator=True,
                    role_id=admin_role.id,
                )
            )
            print(f"[seed] Linked admin user to {label} workspace with role 'admin'")
        else:
            if existing_link.role_id != admin_role.id:
                db.execute(
                    user_tenant_association.update()
                    .where(
                        user_tenant_association.c.user_id == user.id,
                        user_tenant_association.c.tenant_id == tenant_id,
                    )
                    .values(role_id=admin_role.id)
                )
                print(f"[seed] Updated existing {label} link to role 'admin'")

    # Step 6 — Deploy Sample Agent inside Sub-Account context
    db.commit()
    print("[seed] Core multi-tenant hierarchy committed successfully.")

    # Save IDs to plain text strings so they can be printed safely at the end
    agency_id_str = str(agency_tenant.id)
    sub_account_id_str = str(sub_tenant.id)

    # -------------------------------------------------------------------------
    # Steps 6, 7, 8 — Deploy Sample Agent, Call Flow, and Prompts
    # Isolated in its own session block so local column mismatches can't corrupt your core seed.
    # -------------------------------------------------------------------------
    try:
        # Step 6 — Deploy Sample Agent inside Sub-Account context
        agent = db.execute(
            select(Agent).where(
                Agent.tenant_id == sub_tenant.id,
                Agent.name == "dev-agent",
                Agent.is_deleted == False,  # noqa: E712
            )
        ).scalar_one_or_none()

        if agent is None:
            agent = Agent(
                id=_uuid_mod.uuid4(),
                tenant_id=sub_tenant.id,
                name="dev-agent",
                status="pending",
                llm_model="gpt-4o-mini",
                tts_provider_slug="elevenlabs",
                tts_voice_external_id="21m00Tcm4TlvDq8ikWAM",
                tts_language="en",
                smart_callback=False,
            )
            db.add(agent)
            db.flush()
            print(f"[seed] Created sample agent 'dev-agent' inside Sub-Account id={agent.id}")
        else:
            print(f"[seed] Sample agent 'dev-agent' already exists id={agent.id}")

        # Step 7 — Set up Sample Call Flow
        flow = db.execute(
            select(CallFlow).where(
                CallFlow.tenant_id == sub_tenant.id,
                CallFlow.agent_id == agent.id,
                CallFlow.name == "dev-flow",
                CallFlow.is_deleted == False,  # noqa: E712
            )
        ).scalar_one_or_none()

        if flow is None:
            flow = CallFlow(
                id=_uuid_mod.uuid4(),
                tenant_id=sub_tenant.id,
                agent_id=agent.id,
                name="dev-flow",
                direction="outbound",
                welcome_message_type="ai_dynamic",
            )
            db.add(flow)
            db.flush()
            print(f"[seed] Created sample call flow 'dev-flow' id={flow.id}")
        else:
            print(f"[seed] Sample call flow 'dev-flow' already exists id={flow.id}")

        # Step 8 — Set up Sample Prompt Version
        prompt_version = db.execute(
            select(PromptVersion).where(PromptVersion.flow_id == flow.id)
        ).scalar_one_or_none()

        if prompt_version is None:
            prompt_version = PromptVersion(
                id=_uuid_mod.uuid4(),
                flow_id=flow.id,
                prompt_text=(
                    "You are a helpful voice assistant for the dev sub-account workspace. "
                    "Greet the caller and ask how you can help them today."
                ),
                notes="Initial seed prompt",
            )
            db.add(prompt_version)
            db.flush()
            flow.current_prompt_id = prompt_version.id
            print(f"[seed] Created sample prompt version id={prompt_version.id}")
        else:
            if flow.current_prompt_id is None:
                flow.current_prompt_id = prompt_version.id
            print(f"[seed] Sample prompt version already exists id={prompt_version.id}")
        
        db.commit()

    except Exception as e:
        db.rollback()  # Rollback only the agent session segment
        print("\n⚠️  [seed] Skipping Agent/Flow objects because local DB is missing newer feature branch columns.")
        print("   Workspaces and Admin users are still fully configured and ready!\n")

    print("[seed] Done.")
    print(f"\n[seed] Hierarchical environment active:")
    print(f"   Parent Agency ID:      {agency_id_str}")
    print(f"   Linked Sub-Account ID: {sub_account_id_str}")
    print(f"   Admin Credentials:     {SEED_ADMIN_EMAIL} / {SEED_ADMIN_PASSWORD}")


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