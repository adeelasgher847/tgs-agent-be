"""Add candidate status enum to resume table."""

from alembic import op

revision = "20260420_resume_candidate_status"
down_revision = "20260420_schedcall_resint"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_type WHERE typname = 'candidatestatus'
            ) THEN
                CREATE TYPE candidatestatus AS ENUM (
                    'qualified',
                    'partially qualified',
                    'rejected'
                );
            END IF;
        END $$;
        """
    )
    op.execute(
        """
        ALTER TABLE resume
        ADD COLUMN IF NOT EXISTS candidate_status candidatestatus NULL;
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_resume_candidate_status
        ON resume (candidate_status);
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_resume_candidate_status;")
    op.execute(
        """
        ALTER TABLE resume
        DROP COLUMN IF EXISTS candidate_status;
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_type WHERE typname = 'candidatestatus'
            ) THEN
                DROP TYPE candidatestatus;
            END IF;
        END $$;
        """
    )
