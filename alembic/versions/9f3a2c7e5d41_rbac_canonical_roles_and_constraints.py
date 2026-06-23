"""RBAC hardening: canonical 5-tier roles, owner/member retirement, uniqueness

Revision ID: 9f3a2c7e5d41
Revises: 8b1bbaf10244
Create Date: 2026-06-22 10:00:00.000000

Reuses the existing `role` (catalog) + `user_tenant_association` (per-workspace
assignment) tables instead of introducing a new `rbac_roles` table — see
docs/rbac-matrix.md for the design rationale.

Steps (order matters — data must be remapped before the new CHECK constraint
is added, otherwise the constraint would reject the still-present legacy rows):

  1. Drop the old `ck_role_name_valid` constraint (owner/admin/member/config/readonly).
  2. Insert the two net-new canonical roles: manager, billing_only.
  3. Rename: config -> config_only, readonly -> read_only.
  4. Remap user_tenant_association.role_id: owner -> admin, member -> config_only.
     (Workspace creators keep their `is_creator` flag regardless — that flag,
     not the role name, is what grants the permanent admin override.)
  5. Delete the now-unreferenced 'owner' and 'member' catalog rows.
  6. De-duplicate user_tenant_association on (user_id, tenant_id) — no unique
     constraint existed before this migration, so a defensive cleanup runs
     before adding one (keeps the creator row, else the row with a role
     assigned, else the most recently inserted).
  7. Add the uq_user_tenant_association_user_tenant unique constraint.
  8. Re-add ck_role_name_valid scoped to the 5 canonical names.
  9. Drop `role.role` / `role.deleted_at` — unmapped leftovers from the
     abandoned rbac_roles attempt (already removed from the ORM model in
     commit 0d1d97d; no code references them).

Downgrade is best-effort: it restores the old constraint and re-inserts
owner/member rows, but does not attempt to reverse the role_id remap (which
rows were originally 'owner' vs born 'admin' is not recoverable) or undo the
de-duplication of user_tenant_association. This matches the precedent set by
other lossy downgrades in this migration chain (e.g. 166509c9a980).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9f3a2c7e5d41"
down_revision: Union[str, Sequence[str], None] = "8b1bbaf10244"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_OLD_CHECK_NAME = "ck_role_name_valid"
_OLD_VALID_ROLES = ("owner", "admin", "member", "config", "readonly")
_NEW_VALID_ROLES = ("admin", "manager", "config_only", "read_only", "billing_only")
_UNIQUE_NAME = "uq_user_tenant_association_user_tenant"


def upgrade() -> None:
    # 1. Drop the old constraint so data can be remapped freely.
    op.execute(f"ALTER TABLE role DROP CONSTRAINT IF EXISTS {_OLD_CHECK_NAME}")

    # 2. Net-new roles (idempotent).
    op.execute(
        sa.text(
            """
            INSERT INTO role (id, name, description, created_at)
            VALUES
                (gen_random_uuid(), 'manager', 'Mid-tier: full operational access, cannot manage members or billing', NOW()),
                (gen_random_uuid(), 'billing_only', 'Access limited to billing endpoints (usage, pricing)', NOW())
            ON CONFLICT (name) DO NOTHING
            """
        )
    )

    # 3. Renames.
    op.execute("UPDATE role SET name = 'config_only' WHERE name = 'config'")
    op.execute("UPDATE role SET name = 'read_only' WHERE name = 'readonly'")

    # 4. Remap owner -> admin, member -> config_only on the association rows.
    op.execute(
        """
        UPDATE user_tenant_association uta
        SET role_id = (SELECT id FROM role WHERE name = 'admin')
        WHERE uta.role_id = (SELECT id FROM role WHERE name = 'owner')
        """
    )
    op.execute(
        """
        UPDATE user_tenant_association uta
        SET role_id = (SELECT id FROM role WHERE name = 'config_only')
        WHERE uta.role_id = (SELECT id FROM role WHERE name = 'member')
        """
    )

    # 5. Retire the now-unreferenced legacy roles.
    op.execute("DELETE FROM role WHERE name IN ('owner', 'member')")

    # 6. De-duplicate (user_id, tenant_id) before the unique constraint can be added.
    #    Keep: is_creator=True first, then a row with a role assigned, then the
    #    most recently created row (ctid as a stable-enough tiebreaker).
    op.execute(
        """
        DELETE FROM user_tenant_association uta
        WHERE uta.ctid NOT IN (
            SELECT keep.ctid FROM (
                SELECT DISTINCT ON (user_id, tenant_id) ctid
                FROM user_tenant_association
                ORDER BY user_id, tenant_id,
                         is_creator DESC,
                         (role_id IS NOT NULL) DESC,
                         ctid DESC
            ) keep
        )
        """
    )

    # 7. Enforce one association row per (user, tenant) going forward.
    op.create_unique_constraint(
        _UNIQUE_NAME, "user_tenant_association", ["user_id", "tenant_id"]
    )

    # 8. Re-add the CHECK constraint scoped to the 5 canonical roles.
    roles_sql = ", ".join(f"'{r}'" for r in _NEW_VALID_ROLES)
    op.create_check_constraint(
        _OLD_CHECK_NAME, "role", sa.text(f"name IN ({roles_sql})")
    )

    # 9. Drop dead columns left over from the abandoned rbac_roles attempt.
    op.execute("ALTER TABLE role DROP COLUMN IF EXISTS role")
    op.execute("ALTER TABLE role DROP COLUMN IF EXISTS deleted_at")


def downgrade() -> None:
    op.execute(f"ALTER TABLE role DROP CONSTRAINT IF EXISTS {_OLD_CHECK_NAME}")

    op.add_column("role", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("role", sa.Column("role", sa.String(), nullable=True))

    try:
        op.drop_constraint(_UNIQUE_NAME, "user_tenant_association", type_="unique")
    except Exception:
        pass

    op.execute(
        sa.text(
            """
            INSERT INTO role (id, name, description, created_at)
            VALUES
                (gen_random_uuid(), 'owner', 'Owner role with full access to tenant', NOW()),
                (gen_random_uuid(), 'member', 'Regular member role with limited access', NOW())
            ON CONFLICT (name) DO NOTHING
            """
        )
    )
    op.execute("UPDATE role SET name = 'config' WHERE name = 'config_only'")
    op.execute("UPDATE role SET name = 'readonly' WHERE name = 'read_only'")
    op.execute("DELETE FROM role WHERE name IN ('manager', 'billing_only')")

    roles_sql = ", ".join(f"'{r}'" for r in _OLD_VALID_ROLES)
    op.create_check_constraint(
        _OLD_CHECK_NAME, "role", sa.text(f"name IN ({roles_sql})")
    )
