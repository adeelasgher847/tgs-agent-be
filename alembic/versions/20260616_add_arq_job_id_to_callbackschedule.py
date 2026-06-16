"""add arq_job_id to callbackschedule

Revision ID: 20260616_callback_arq
Revises: 20260616_analytics_index
Create Date: 2026-06-16

Tracks which ARQ Redis job ID corresponds to each pending callback row.
Allows the recovery cron to skip already-enqueued rows and provides
observability into the ARQ ↔ PostgreSQL state correspondence.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260616_callback_arq"
down_revision: Union[str, None] = "20260616_analytics_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "callbackschedule",
        sa.Column("arq_job_id", sa.String(255), nullable=True),
    )
    op.create_index(
        "ix_callbackschedule_arq_job_id",
        "callbackschedule",
        ["arq_job_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_callbackschedule_arq_job_id", table_name="callbackschedule")
    op.drop_column("callbackschedule", "arq_job_id")
