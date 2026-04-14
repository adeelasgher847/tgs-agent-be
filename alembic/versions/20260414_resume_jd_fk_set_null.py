"""Set resume.job_description_id FK to ON DELETE SET NULL."""

from alembic import op

revision = "20260414_resume_jd_setnull"
down_revision = "20260414_resume_match"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE resume
        DROP CONSTRAINT IF EXISTS resume_job_description_id_fkey;
        """
    )
    op.execute(
        """
        ALTER TABLE resume
        ADD CONSTRAINT resume_job_description_id_fkey
        FOREIGN KEY (job_description_id)
        REFERENCES jobdescription(id)
        ON DELETE SET NULL;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE resume
        DROP CONSTRAINT IF EXISTS resume_job_description_id_fkey;
        """
    )
    op.execute(
        """
        ALTER TABLE resume
        ADD CONSTRAINT resume_job_description_id_fkey
        FOREIGN KEY (job_description_id)
        REFERENCES jobdescription(id);
        """
    )
