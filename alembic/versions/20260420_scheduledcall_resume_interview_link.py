"""Link scheduledcall rows to resume interviews."""

from alembic import op

revision = "20260420_schedcall_resint"
down_revision = "20260416_resume_interviews"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE scheduledcall
        ADD COLUMN IF NOT EXISTS resume_interview_id uuid NULL;
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM information_schema.table_constraints
                WHERE constraint_name = 'fk_scheduledcall_resume_interview_id'
            ) THEN
                ALTER TABLE scheduledcall
                ADD CONSTRAINT fk_scheduledcall_resume_interview_id
                FOREIGN KEY (resume_interview_id)
                REFERENCES resumeinterview(id)
                ON DELETE CASCADE;
            END IF;
        END $$;
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_scheduledcall_resume_interview_id ON scheduledcall (resume_interview_id);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_scheduledcall_resume_interview_id;")
    op.execute(
        """
        ALTER TABLE scheduledcall
        DROP CONSTRAINT IF EXISTS fk_scheduledcall_resume_interview_id;
        """
    )
    op.execute(
        """
        ALTER TABLE scheduledcall
        DROP COLUMN IF EXISTS resume_interview_id;
        """
    )
