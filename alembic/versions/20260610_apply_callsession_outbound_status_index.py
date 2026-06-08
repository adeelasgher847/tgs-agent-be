"""apply_callsession_outbound_status_index

Linear follow-up: the outbound status index branch was skipped when the
recording merge ran against the duplicate a1b2c3d4e5f6 revision id.

Revision ID: 20260610_outbound_idx
Revises: 20260609_call_recording
Create Date: 2026-06-10
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260610_outbound_idx"
down_revision: Union[str, Sequence[str], None] = "20260609_call_recording"
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
    conn = op.get_bind()
    if _has_index(conn, "ix_callsession_tenant_calltype_status"):
        op.drop_index("ix_callsession_tenant_calltype_status", table_name="callsession")
