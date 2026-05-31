"""Add years_experience_max to jobdescription."""

from alembic import op

revision = "20260413_jd_exp_max"
down_revision = "20260402_resume_batch_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE jobdescription
        ADD COLUMN IF NOT EXISTS years_experience_max integer NULL;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE jobdescription
        DROP COLUMN IF EXISTS years_experience_max;
        """
    )
