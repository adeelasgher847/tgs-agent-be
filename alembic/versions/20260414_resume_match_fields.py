"""Add persisted match fields to resume."""

from alembic import op

revision = "20260414_resume_match"
down_revision = "20260414_jd_match_thr"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE resume
        ADD COLUMN IF NOT EXISTS overall_match_score double precision NULL,
        ADD COLUMN IF NOT EXISTS match_percent integer NULL,
        ADD COLUMN IF NOT EXISTS fit_label varchar(32) NULL,
        ADD COLUMN IF NOT EXISTS is_relevant boolean NULL;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE resume
        DROP COLUMN IF EXISTS is_relevant,
        DROP COLUMN IF EXISTS fit_label,
        DROP COLUMN IF EXISTS match_percent,
        DROP COLUMN IF EXISTS overall_match_score;
        """
    )
