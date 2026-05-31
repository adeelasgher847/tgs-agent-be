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
