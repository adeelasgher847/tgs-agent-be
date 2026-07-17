"""add amd voicemail fields to batchjob

Revision ID: 20260716_amd_voicemail
Revises: b2d3c4e5f6a8
Create Date: 2026-07-16

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20260716_amd_voicemail"
down_revision: Union[str, Sequence[str], None] = "b2d3c4e5f6a8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "batchjob",
        sa.Column("voicemail_action", sa.Text(), nullable=False, server_default="skip"),
    )
    op.add_column(
        "batchjob",
        sa.Column("voicemail_message", sa.Text(), nullable=True),
    )
    op.add_column(
        "batchjob",
        sa.Column("voicemail_skipped_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "batchjob",
        sa.Column("voicemail_message_left_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_check_constraint(
        "ck_batchjob_voicemail_action",
        "batchjob",
        "voicemail_action IN ('skip', 'leave_message', 'continue')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_batchjob_voicemail_action", "batchjob", type_="check")
    op.drop_column("batchjob", "voicemail_message_left_count")
    op.drop_column("batchjob", "voicemail_skipped_count")
    op.drop_column("batchjob", "voicemail_message")
    op.drop_column("batchjob", "voicemail_action")
