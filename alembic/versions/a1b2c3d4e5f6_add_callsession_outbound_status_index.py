"""add_callsession_outbound_status_index

Adds a composite index on callsession(tenant_id, call_type, status) to support
efficient per-workspace concurrent outbound call count queries.

Revision ID: 20260608_outbound_status_idx
Revises: f6b7c8d9e0f1
Create Date: 2026-06-08
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260608_outbound_status_idx"
down_revision: Union[str, Sequence[str], None] = "f6b7c8d9e0f1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_index(conn, name: str) -> bool:
    return conn.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM pg_indexes "
            "WHERE schemaname = 'public' AND indexname = :name)"
        ),
        {"name": name},
    ).scalar()


def upgrade() -> None:
    conn = op.get_bind()
    if not _has_index(conn, "ix_callsession_tenant_calltype_status"):
        op.create_index(
            "ix_callsession_tenant_calltype_status",
            "callsession",
            ["tenant_id", "call_type", "status"],
            unique=False,
        )


def downgrade() -> None:
    op.drop_index("ix_callsession_tenant_calltype_status", table_name="callsession")
