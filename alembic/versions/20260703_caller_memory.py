"""Add cross-session caller memory columns and lookup index

Revision ID: 20260703_caller_memory
Revises: 20260702_ab_prompt_testing
Create Date: 2026-07-03 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "20260703_caller_memory"
down_revision: Union[str, Sequence[str], None] = "20260702_ab_prompt_testing"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "callflow",
        sa.Column(
            "caller_memory_enabled",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
    )
    op.add_column(
        "callflow",
        sa.Column(
            "caller_memory_window",
            sa.Integer(),
            nullable=False,
            server_default="3",
        ),
    )
    op.create_check_constraint(
        "ck_callflow_caller_memory_window",
        "callflow",
        "caller_memory_window >= 1 AND caller_memory_window <= 10",
    )

    op.add_column(
        "callsession",
        sa.Column("transcript_summary", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_callsession_memory_lookup",
        "callsession",
        ["tenant_id", "call_flow_id", "from_number", "start_time"],
    )


def downgrade() -> None:
    op.drop_index("ix_callsession_memory_lookup", table_name="callsession")
    op.drop_column("callsession", "transcript_summary")

    op.drop_constraint("ck_callflow_caller_memory_window", "callflow", type_="check")
    op.drop_column("callflow", "caller_memory_window")
    op.drop_column("callflow", "caller_memory_enabled")
