"""Add job_description_id to resume."""

from alembic import op

revision = "20260413_resume_jd_fk"
down_revision = "20260413_jd_exp_max"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE resume
        ADD COLUMN IF NOT EXISTS job_description_id uuid NULL REFERENCES jobdescription(id);
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_resume_job_description_id ON resume(job_description_id);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_resume_job_description_id;")
    op.execute(
        """
        ALTER TABLE resume
        DROP COLUMN IF EXISTS job_description_id;
        """
    )
