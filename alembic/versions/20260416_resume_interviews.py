"""Add resume interview tracking tables."""

from alembic import op

revision = "20260416_resume_interviews"
down_revision = "20260414_resume_jd_setnull"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS resumeinterview (
            id uuid PRIMARY KEY,
            tenant_id uuid NOT NULL REFERENCES tenant(id),
            resume_id uuid NOT NULL REFERENCES resume(id),
            job_description_id uuid NULL REFERENCES jobdescription(id) ON DELETE SET NULL,
            agent_id uuid NOT NULL REFERENCES agent(id),
            call_session_id uuid NULL REFERENCES callsession(id) ON DELETE SET NULL,
            candidate_phone varchar(64) NOT NULL,
            scheduled_at timestamptz NOT NULL,
            status varchar(40) NOT NULL DEFAULT 'SCHEDULE_REQUESTED',
            crm_type varchar(20) NULL,
            crm_item_id varchar(128) NULL,
            crm_batch_id varchar(128) NULL,
            phone_number_id uuid NULL REFERENCES phonenumber(id),
            twilio_call_sid varchar(255) NULL,
            attempt_count integer NOT NULL DEFAULT 0,
            last_error text NULL,
            metadata_json jsonb NULL,
            created_by uuid NOT NULL REFERENCES "user"(id),
            updated_by uuid NOT NULL REFERENCES "user"(id),
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NULL
        );
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_resumeinterview_id ON resumeinterview (id);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_resumeinterview_tenant_id ON resumeinterview (tenant_id);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_resumeinterview_resume_id ON resumeinterview (resume_id);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_resumeinterview_job_description_id ON resumeinterview (job_description_id);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_resumeinterview_agent_id ON resumeinterview (agent_id);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_resumeinterview_call_session_id ON resumeinterview (call_session_id);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_resumeinterview_status ON resumeinterview (status);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_resumeinterview_crm_item_id ON resumeinterview (crm_item_id);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_resumeinterview_crm_batch_id ON resumeinterview (crm_batch_id);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_resumeinterview_twilio_call_sid ON resumeinterview (twilio_call_sid);")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS resumeinterviewevent (
            id uuid PRIMARY KEY,
            tenant_id uuid NOT NULL REFERENCES tenant(id),
            resume_interview_id uuid NOT NULL REFERENCES resumeinterview(id) ON DELETE CASCADE,
            event_type varchar(80) NOT NULL,
            event_payload jsonb NULL,
            created_by uuid NOT NULL REFERENCES "user"(id),
            created_at timestamptz NOT NULL DEFAULT now()
        );
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_resumeinterviewevent_id ON resumeinterviewevent (id);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_resumeinterviewevent_tenant_id ON resumeinterviewevent (tenant_id);")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_resumeinterviewevent_resume_interview_id ON resumeinterviewevent (resume_interview_id);"
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_resumeinterviewevent_event_type ON resumeinterviewevent (event_type);")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS resumeinterviewevent;")
    op.execute("DROP TABLE IF EXISTS resumeinterview;")
