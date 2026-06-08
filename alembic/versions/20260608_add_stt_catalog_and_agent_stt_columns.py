"""add STT catalog tables and agent STT columns

Revision ID: 20260608_stt_catalog
Revises: 20260602_schema_v2
Create Date: 2026-06-08

Creates sttprovider + sttmodel catalog tables, agent STT FK columns,
and seeds deepgram (nova-3, nova-2), google (chirp-3), elevenlabs (scrib-v1).
"""
from typing import Sequence, Union

import json
import uuid as _uuid

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "20260608_stt_catalog"
down_revision: Union[str, Sequence[str], None] = "20260602_schema_v2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(conn, name: str) -> bool:
    return (
        conn.execute(
            sa.text("SELECT to_regclass(:name)"),
            {"name": name},
        ).scalar()
        is not None
    )


def _column_exists(conn, table: str, column: str) -> bool:
    return (
        conn.execute(
            sa.text(
                """
                SELECT 1 FROM information_schema.columns
                WHERE table_name = :table AND column_name = :column
                LIMIT 1
                """
            ),
            {"table": table, "column": column},
        ).scalar()
        is not None
    )


def upgrade() -> None:
    conn = op.get_bind()

    if not _table_exists(conn, "sttprovider"):
        op.create_table(
            "sttprovider",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
            sa.Column("slug", sa.String(50), nullable=False, unique=True),
            sa.Column("display_name", sa.String(100), nullable=False),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
            sa.Column("supports_streaming", sa.Boolean(), nullable=False, server_default=sa.text("true")),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index("ix_sttprovider_slug", "sttprovider", ["slug"], unique=True)

    if not _table_exists(conn, "sttmodel"):
        op.create_table(
            "sttmodel",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
            sa.Column(
                "provider_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("sttprovider.id"),
                nullable=False,
            ),
            sa.Column("external_model_id", sa.String(255), nullable=False),
            sa.Column("display_name", sa.String(255), nullable=False),
            sa.Column("language_code", sa.String(20), nullable=False, server_default="en-US"),
            sa.Column("sample_rate_hz", sa.Integer(), nullable=False, server_default="16000"),
            sa.Column("encoding", sa.String(20), nullable=False, server_default="LINEAR16"),
            sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.UniqueConstraint("provider_id", "external_model_id", name="uq_sttmodel_provider_external"),
        )
        op.create_index("ix_sttmodel_provider_id", "sttmodel", ["provider_id"])
        op.create_index("ix_sttmodel_is_active", "sttmodel", ["is_active"])

    if not _column_exists(conn, "agent", "stt_provider_slug"):
        op.add_column("agent", sa.Column("stt_provider_slug", sa.String(40), nullable=True))
    if not _column_exists(conn, "agent", "stt_model_external_id"):
        op.add_column("agent", sa.Column("stt_model_external_id", sa.String(255), nullable=True))
    if not _column_exists(conn, "agent", "stt_language_code"):
        op.add_column("agent", sa.Column("stt_language_code", sa.String(20), nullable=True))
    if not _column_exists(conn, "agent", "stt_provider_id"):
        op.add_column(
            "agent",
            sa.Column(
                "stt_provider_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("sttprovider.id", ondelete="SET NULL"),
                nullable=True,
            ),
        )
    if not _column_exists(conn, "agent", "stt_model_id"):
        op.add_column(
            "agent",
            sa.Column(
                "stt_model_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("sttmodel.id", ondelete="SET NULL"),
                nullable=True,
            ),
        )
    if not _column_exists(conn, "agent", "stt_settings_json"):
        op.add_column(
            "agent",
            sa.Column("stt_settings_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        )

    # Agent indexes (idempotent)
    op.execute(
        sa.text(
            "CREATE INDEX IF NOT EXISTS ix_agent_stt_provider_id ON agent (stt_provider_id)"
        )
    )
    op.execute(
        sa.text("CREATE INDEX IF NOT EXISTS ix_agent_stt_model_id ON agent (stt_model_id)")
    )

    deepgram_id = str(_uuid.UUID("00000000-0000-0000-0000-000000000101"))
    google_id = str(_uuid.UUID("00000000-0000-0000-0000-000000000102"))
    elevenlabs_id = str(_uuid.UUID("00000000-0000-0000-0000-000000000103"))

    op.execute(
        sa.text(
            """
            INSERT INTO sttprovider (id, slug, display_name, is_active, supports_streaming)
            VALUES
              (:did, 'deepgram', 'Deepgram', true, true),
              (:gid, 'google', 'Google Cloud STT', true, true),
              (:eid, 'elevenlabs', 'ElevenLabs Scribe', true, false)
            ON CONFLICT (slug) DO NOTHING
            """
        ).bindparams(did=deepgram_id, gid=google_id, eid=elevenlabs_id)
    )

    nova3_id = str(_uuid.UUID("00000000-0000-0000-0001-000000000001"))
    nova2_id = str(_uuid.UUID("00000000-0000-0000-0001-000000000002"))
    chirp3_id = str(_uuid.UUID("00000000-0000-0000-0001-000000000003"))
    scribe_id = str(_uuid.UUID("00000000-0000-0000-0001-000000000004"))

    model_rows = [
        (
            nova3_id,
            deepgram_id,
            "nova-3",
            "Deepgram Nova 3",
            "en",
            8000,
            "MULAW",
            json.dumps(
                {"api_model": "nova-3", "encoding": "mulaw", "sample_rate_hz": 8000}
            ),
        ),
        (
            nova2_id,
            deepgram_id,
            "nova-2",
            "Deepgram Nova 2",
            "en",
            8000,
            "MULAW",
            json.dumps(
                {"api_model": "nova-2", "encoding": "mulaw", "sample_rate_hz": 8000}
            ),
        ),
        (
            chirp3_id,
            google_id,
            "chirp-3",
            "Google Chirp 3",
            "en-AU",
            16000,
            "LINEAR16",
            json.dumps(
                {
                    "api_model": "chirp_3",
                    "google_model": "phone_call",
                    "use_enhanced": True,
                    "encoding": "LINEAR16",
                    "sample_rate_hz": 16000,
                }
            ),
        ),
        (
            scribe_id,
            elevenlabs_id,
            "scrib-v1",
            "ElevenLabs Scribe",
            "en",
            16000,
            "LINEAR16",
            json.dumps({"api_model": "scribe_v1"}),
        ),
    ]
    for row in model_rows:
        conn.execute(
            sa.text(
                """
                INSERT INTO sttmodel (
                  id, provider_id, external_model_id, display_name,
                  language_code, sample_rate_hz, encoding, metadata_json, is_active
                )
                VALUES (
                  CAST(:id AS uuid), CAST(:provider_id AS uuid), :external_model_id,
                  :display_name, :language_code, :sample_rate_hz, :encoding,
                  CAST(:metadata_json AS jsonb), true
                )
                ON CONFLICT (provider_id, external_model_id) DO NOTHING
                """
            ),
            {
                "id": row[0],
                "provider_id": row[1],
                "external_model_id": row[2],
                "display_name": row[3],
                "language_code": row[4],
                "sample_rate_hz": row[5],
                "encoding": row[6],
                "metadata_json": row[7],
            },
        )

    op.execute(
        sa.text(
            """
            UPDATE agent
            SET
              stt_provider_slug      = 'deepgram',
              stt_model_external_id = 'nova-3',
              stt_language_code    = 'en',
              stt_provider_id      = :pid,
              stt_model_id         = :mid
            WHERE stt_provider_slug IS NULL
            """
        ).bindparams(pid=deepgram_id, mid=nova3_id)
    )

    conn.execute(
        sa.text(
            """
            UPDATE sttmodel
            SET metadata_json = (
                COALESCE(metadata_json::jsonb, '{}'::jsonb)
                || CAST(:patch AS jsonb)
            )::json
            WHERE external_model_id = 'chirp-3'
              AND provider_id IN (SELECT id FROM sttprovider WHERE slug = 'google')
            """
        ),
        {
            "patch": json.dumps({"use_enhanced": True, "google_model": "phone_call"}),
        },
    )


def downgrade() -> None:
    op.drop_index("ix_agent_stt_model_id", table_name="agent")
    op.drop_index("ix_agent_stt_provider_id", table_name="agent")
    op.drop_column("agent", "stt_settings_json")
    op.drop_column("agent", "stt_model_id")
    op.drop_column("agent", "stt_provider_id")
    op.drop_column("agent", "stt_language_code")
    op.drop_column("agent", "stt_model_external_id")
    op.drop_column("agent", "stt_provider_slug")

    op.drop_index("ix_sttmodel_is_active", table_name="sttmodel")
    op.drop_index("ix_sttmodel_provider_id", table_name="sttmodel")
    op.drop_table("sttmodel")

    op.drop_index("ix_sttprovider_slug", table_name="sttprovider")
    op.drop_table("sttprovider")
