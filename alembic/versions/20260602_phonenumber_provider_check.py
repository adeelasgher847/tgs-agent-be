"""add phonenumber provider check constraint

Revision ID: 20260602_phonenumber_provider
Revises: 20260601_callflow_schema
Create Date: 2026-06-02

Enforce provider IN ('twilio', 'external') at the DB layer.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260602_phonenumber_provider"
down_revision: Union[str, Sequence[str], None] = "20260601_callflow_schema"
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

    if not _has_constraint(conn, "phonenumber", "ck_phonenumber_provider"):
        op.create_check_constraint(
            "ck_phonenumber_provider",
            "phonenumber",
            "provider IN ('twilio', 'external')",
        )


def downgrade() -> None:
    conn = op.get_bind()

    if _has_constraint(conn, "phonenumber", "ck_phonenumber_provider"):
        op.drop_constraint("ck_phonenumber_provider", "phonenumber", type_="check")
