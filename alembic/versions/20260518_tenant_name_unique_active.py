"""unique tenant name among active (non-deleted) workspaces

Revision ID: 20260518_tenant_name_uq
Revises: 20260518_tenant_deleted_at
Create Date: 2026-05-18 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260518_tenant_name_uq"
down_revision: Union[str, Sequence[str], None] = "20260518_tenant_deleted_at"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Rename duplicate active names (keep oldest row per name) so the unique index can be applied.
    op.execute(
        sa.text(
            """
            WITH ranked AS (
                SELECT
                    id,
                    name,
                    ROW_NUMBER() OVER (
                        PARTITION BY name
                        ORDER BY created_at ASC NULLS LAST, id ASC
                    ) AS rn
                FROM tenant
                WHERE deleted_at IS NULL
            )
            UPDATE tenant AS t
            SET name = LEFT(r.name, 42) || ' #' || ranked.rn::text
            FROM ranked
            JOIN tenant AS r ON r.id = ranked.id
            WHERE t.id = ranked.id
              AND ranked.rn > 1
            """
        )
    )

    op.create_index(
        "uq_tenant_name_active",
        "tenant",
        ["name"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_tenant_name_active", table_name="tenant")
