"""Add config and readonly roles; CHECK constraint on role.name

Revision ID: 20260521_role_config_readonly
Revises: 20260521_schema_v1_gaps
Create Date: 2026-05-21 12:00:00.000000

Adds two new roles required by schema v1:
  - config   : can configure workspace settings; cannot manage users
  - readonly  : read-only access; blocked from mutating endpoints

Also adds a CHECK constraint on role.name so only the five canonical values
are accepted: owner, admin, member, config, readonly.

Run:
    alembic upgrade head
Revert:
    alembic downgrade -1
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260521_role_config_readonly"
down_revision: Union[str, Sequence[str], None] = "20260521_schema_v1_gaps"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_VALID_ROLES = ("owner", "admin", "member", "config", "readonly")
_CHECK_NAME = "ck_role_name_valid"


def upgrade() -> None:
    # Insert new roles if they don't already exist (idempotent)
    op.execute(
        sa.text(
            """
            INSERT INTO role (id, name, description, created_at)
            VALUES
                (gen_random_uuid(), 'config',   'Configure workspace settings; cannot manage users', NOW()),
                (gen_random_uuid(), 'readonly',  'Read-only access; blocked from mutating endpoints',  NOW())
            ON CONFLICT (name) DO NOTHING
            """
        )
    )

    # Add CHECK constraint so the DB rejects invalid role names
    roles_sql = ", ".join(f"'{r}'" for r in _VALID_ROLES)
    op.create_check_constraint(
        _CHECK_NAME,
        "role",
        sa.text(f"name IN ({roles_sql})"),
    )


def downgrade() -> None:
    op.drop_constraint(_CHECK_NAME, "role", type_="check")

    # Remove the two new roles only when no user_tenant_association references them.
    op.execute(
        sa.text(
            """
            DELETE FROM role
            WHERE name IN ('config', 'readonly')
              AND id NOT IN (
                  SELECT DISTINCT role_id
                  FROM user_tenant_association
                  WHERE role_id IS NOT NULL
              )
            """
        )
    )
