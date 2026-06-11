"""Replace Pinecone-backed KB tables with pgvector-native schema

Revision ID: 20260611_kb_pgvector
Revises: 20260610_webhooks
Create Date: 2026-06-11 00:00:00.000000
"""

revision = "20260611_kb_pgvector"
down_revision = "20260610_webhooks"
branch_labels = None
depends_on = None

from alembic import op


def upgrade() -> None:
    # Drop old Pinecone-backed tables (chunk first — FK constraint)
    op.execute("DROP TABLE IF EXISTS knowledgebasechunk CASCADE;")
    op.execute("DROP TABLE IF EXISTS knowledgebasedocument CASCADE;")

    # Enable pgvector
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # knowledge_bases: one KB per workspace (or many)
    op.execute(
        """
        CREATE TABLE knowledge_bases (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id UUID NOT NULL REFERENCES tenant(id) ON DELETE CASCADE,
            name        TEXT NOT NULL,
            description TEXT,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at  TIMESTAMPTZ
        );
        """
    )
    op.execute(
        "CREATE INDEX ix_knowledge_bases_workspace_id ON knowledge_bases(workspace_id);"
    )

    # kb_files: tracks uploaded files per KB
    op.execute(
        """
        CREATE TABLE kb_files (
            id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            kb_id             UUID NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
            original_filename TEXT NOT NULL,
            size_bytes        BIGINT,
            file_type         TEXT,
            gcs_path          TEXT,
            status            TEXT NOT NULL DEFAULT 'processing'
                                  CHECK (status IN ('processing', 'ready', 'error')),
            error_message     TEXT,
            chunk_count       INT,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )
    op.execute("CREATE INDEX ix_kb_files_kb_id ON kb_files(kb_id);")

    # kb_chunks: stores text chunks and their 1536-dim embeddings
    op.execute(
        """
        CREATE TABLE kb_chunks (
            id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            kb_id      UUID NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
            file_id    UUID REFERENCES kb_files(id) ON DELETE SET NULL,
            content    TEXT NOT NULL,
            embedding  vector(1536),
            metadata   JSONB NOT NULL DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )
    op.execute("CREATE INDEX ix_kb_chunks_kb_id ON kb_chunks(kb_id);")
    op.execute("CREATE INDEX ix_kb_chunks_file_id ON kb_chunks(file_id);")

    # HNSW index for fast approximate cosine-similarity search
    op.execute(
        """
        CREATE INDEX ix_kb_chunks_embedding_hnsw
            ON kb_chunks USING hnsw (embedding vector_cosine_ops)
            WITH (m=16, ef_construction=64);
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS kb_chunks CASCADE;")
    op.execute("DROP TABLE IF EXISTS kb_files CASCADE;")
    op.execute("DROP TABLE IF EXISTS knowledge_bases CASCADE;")
    # Restore old tables
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
