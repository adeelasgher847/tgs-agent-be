from alembic import op

revision = "20260318_kb_tables"
down_revision = "d9c017b24cf7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS knowledgebasedocument (
            id uuid PRIMARY KEY,
            tenant_id uuid NOT NULL REFERENCES tenant(id),
            agent_id uuid REFERENCES agent(id),
            title varchar(255) NOT NULL,
            source_type varchar(120) NOT NULL,
            source_ref varchar(512) NOT NULL,
            version varchar(120) NOT NULL DEFAULT 'v1',
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NULL,
            is_active boolean NOT NULL DEFAULT true
        );
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS knowledgebasechunk (
            id uuid PRIMARY KEY,
            document_id uuid NOT NULL REFERENCES knowledgebasedocument(id),
            chunk_index integer NOT NULL,
            vector_id varchar(300) NOT NULL UNIQUE,
            text_preview varchar(700) NULL,
            created_at timestamptz NOT NULL DEFAULT now()
        );
        """
    )

    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_knowledgebasedocument_tenant_id ON knowledgebasedocument(tenant_id);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_knowledgebasedocument_agent_id ON knowledgebasedocument(agent_id);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_knowledgebasechunk_document_id ON knowledgebasechunk(document_id);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_knowledgebasechunk_chunk_index ON knowledgebasechunk(chunk_index);"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS knowledgebasechunk;")
    op.execute("DROP TABLE IF EXISTS knowledgebasedocument;")

