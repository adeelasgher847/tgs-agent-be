"""callflow updated_at default and direction check constraint

Revision ID: 20260601_callflow_schema
Revises: 20260529_merge_heads
Create Date: 2026-06-01

- Backfill callflow.updated_at from created_at where NULL
- Set server_default on callflow.updated_at
- Add ck_callflow_direction CHECK constraint if missing
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260601_callflow_schema"
down_revision: Union[str, Sequence[str], None] = "20260529_merge_heads"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_constraint(conn, table: str, constraint: str) -> bool:
    return conn.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.table_constraints "
            "WHERE table_schema = 'public' AND table_name = :tbl AND constraint_name = :con)"
        ),
        {"tbl": table, "con": constraint},
    ).scalar()


def upgrade() -> None:
    conn = op.get_bind()

    op.execute(
        "UPDATE callflow SET updated_at = created_at WHERE updated_at IS NULL"
    )
    op.alter_column(
        "callflow",
        "updated_at",
        server_default=sa.text("now()"),
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=True,
    )

    if not _has_constraint(conn, "callflow", "ck_callflow_direction"):
        op.create_check_constraint(
            "ck_callflow_direction",
            "callflow",
            "direction IN ('inbound', 'outbound')",
        )


def downgrade() -> None:
    conn = op.get_bind()

    if _has_constraint(conn, "callflow", "ck_callflow_direction"):
        op.drop_constraint("ck_callflow_direction", "callflow", type_="check")

    op.alter_column(
        "callflow",
        "updated_at",
        server_default=None,
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=True,
    )
