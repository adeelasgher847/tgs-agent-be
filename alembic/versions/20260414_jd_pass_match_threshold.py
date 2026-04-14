"""Add pass_match_threshold to jobdescription."""

from alembic import op

revision = "20260414_jd_match_thr"
down_revision = "20260413_resume_jd_fk"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE jobdescription
        ADD COLUMN IF NOT EXISTS pass_match_threshold double precision NOT NULL DEFAULT 0.5;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE jobdescription
        DROP COLUMN IF EXISTS pass_match_threshold;
        """
    )
