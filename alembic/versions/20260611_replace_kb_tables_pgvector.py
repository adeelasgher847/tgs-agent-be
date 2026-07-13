"""Migrate kbchunk.embedding from text to vector(1536) and drop legacy Pinecone tables

Revision ID: 20260611_kb_pgvector
Revises: 20260611_cleanup_orphan_tables
Create Date: 2026-06-11 00:00:00.000000
"""

revision = "20260611_kb_pgvector"
down_revision = "20260610_webhooks"
branch_labels = None
depends_on = None

from alembic import op


def upgrade() -> None:
    # Drop legacy Pinecone-backed tables (chunk first — FK constraint)
    op.execute("DROP TABLE IF EXISTS knowledgebasechunk CASCADE;")
    op.execute("DROP TABLE IF EXISTS knowledgebasedocument CASCADE;")

    # Enable pgvector (idempotent)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Cast the existing text column (JSON array string) to vector(1536).
    # pgvector accepts the "[f1,f2,...]" text format that json.dumps produces.
    # NULL values pass through unchanged.
    op.execute(
        """
        ALTER TABLE kbchunk
            ALTER COLUMN embedding TYPE vector(1536)
            USING CASE
                WHEN embedding IS NULL THEN NULL
                ELSE embedding::vector(1536)
            END;
        """
    )

    # HNSW index for fast approximate cosine-similarity search
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_kbchunk_embedding_hnsw
            ON kbchunk USING hnsw (embedding vector_cosine_ops)
            WITH (m=16, ef_construction=64);
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_kbchunk_embedding_hnsw;")

    # Cast vector back to text (pgvector's output format is "[f1,f2,...]")
    op.execute(
        """
        ALTER TABLE kbchunk
            ALTER COLUMN embedding TYPE text
            USING CASE
                WHEN embedding IS NULL THEN NULL
                ELSE embedding::text
            END;
        """
    )

    # Restore legacy Pinecone-backed tables
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
