<<<<<<< HEAD
"""restore missing resume batch id revision

Revision ID: 20260402_resume_batch_id
Revises: b6d5f9e0d2aa
Create Date: 2026-04-02 22:45:00.000000

"""
from typing import Sequence, Union


revision: str = "20260402_resume_batch_id"
down_revision: Union[str, Sequence[str], None] = "b6d5f9e0d2aa"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # This revision restores a missing migration link so existing databases
    # can continue upgrading. It intentionally performs no schema changes.
    pass


def downgrade() -> None:
    pass
=======
"""Add batch_id to resume for batch tracking."""

from alembic import op

revision = "20260402_resume_batch_id"
down_revision = "20260326_job_description_table"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE resume
        ADD COLUMN IF NOT EXISTS batch_id uuid NULL;
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_resume_batch_id ON resume(batch_id);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_resume_batch_id;")
    op.execute(
        """
        ALTER TABLE resume
        DROP COLUMN IF EXISTS batch_id;
        """
    )
>>>>>>> feature/ai-recruit
