"""add_call_recording_gcs_fields

Merges outbound-dispatch and stt-catalog heads, then adds GCS recording
columns to callsession.

Revision ID: 20260609_call_recording
Revises: 20260608_outbound_status_idx, 20260608_stt_catalog
Create Date: 2026-06-09

Changes:
- callsession: recording_gcs_path VARCHAR(500) nullable
- callsession: recording_error BOOLEAN NOT NULL default false
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260609_call_recording"
down_revision: Union[str, Sequence[str], None] = (
    "20260608_outbound_status_idx",
    "20260608_stt_catalog",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(conn, table: str, column: str) -> bool:
    return conn.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = :tbl AND column_name = :col)"
        ),
        {"tbl": table, "col": column},
    ).scalar()


def upgrade() -> None:
    conn = op.get_bind()

    if not _has_column(conn, "callsession", "recording_gcs_path"):
        op.add_column(
            "callsession",
            sa.Column("recording_gcs_path", sa.String(500), nullable=True),
        )

    if not _has_column(conn, "callsession", "recording_error"):
        op.add_column(
            "callsession",
            sa.Column(
                "recording_error",
                sa.Boolean(),
                nullable=False,
                server_default="false",
            ),
        )


def downgrade() -> None:
    conn = op.get_bind()

    if _has_column(conn, "callsession", "recording_error"):
        op.drop_column("callsession", "recording_error")

    if _has_column(conn, "callsession", "recording_gcs_path"):
        op.drop_column("callsession", "recording_gcs_path")
