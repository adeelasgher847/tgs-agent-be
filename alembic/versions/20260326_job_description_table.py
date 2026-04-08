"""Create jobdescription table for recruiter/manual JD workflow."""

from alembic import op

revision = "20260326_job_description_table"
down_revision = "20260318_kb_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS jobdescription (
            id uuid PRIMARY KEY,
            tenant_id uuid NOT NULL REFERENCES tenant(id),

            job_title varchar(255) NOT NULL,
            required_skills jsonb NULL,
            years_experience_min integer NULL,
            education_requirements text NULL,
            location varchar(255) NULL,
            salary_min numeric(12, 2) NULL,
            salary_max numeric(12, 2) NULL,
            currency varchar(12) NULL,
            employment_type varchar(50) NULL,
            key_responsibilities jsonb NULL,
            required_certifications jsonb NULL,

            raw_text text NULL,
            extracted_skills jsonb NULL,
            keywords jsonb NULL,
            skill_weight_matrix jsonb NULL,
            matching_criteria jsonb NULL,
            processing_status varchar(20) NOT NULL DEFAULT 'PENDING',
            version integer NOT NULL DEFAULT 1,

            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NULL,
            created_by uuid NOT NULL REFERENCES "user"(id),
            updated_by uuid NOT NULL REFERENCES "user"(id)
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_jobdescription_tenant_id ON jobdescription(tenant_id);"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS jobdescription;")
