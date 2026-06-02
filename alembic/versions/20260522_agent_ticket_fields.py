"""agent ticket fields: status, llm_model, tts_model trio, byo elevenlabs key

Revision ID: 20260522_agent_ticket
Revises: 20260521_merge_heads
Create Date: 2026-05-22 00:00:00.000000

Adds the fields required by the agent-management ticket so
``/api/v1/agent`` (Voice Agent) endpoints can persist the ticket-shape contract:

  - status                       : 'active' | 'inactive' | 'draft'
  - llm_model                    : string identifier from the LLM allow-list
  - tts_provider_slug            : 'rime' | '11labs' | '11labs_byo'
  - tts_voice_external_id        : provider-side voice id
  - tts_language                 : language code (e.g. 'en', 'es')
  - encrypted_elevenlabs_api_key : encrypted BYO key for 11labs_byo

Also relaxes ``created_by`` / ``updated_by`` to NULLABLE so machine-to-machine
API-key requests (no resolved user) can create / update agents while
existing dashboard JWT flows continue to populate these fields.

Idempotent: safe if re-run (uses information_schema checks).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260522_agent_ticket"
down_revision: Union[str, Sequence[str], None] = "20260521_merge_heads"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(conn, table: str, column: str) -> bool:
    return conn.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = :tbl AND column_name = :col)"
        ),
        {"tbl": table, "col": column},
    ).scalar()


def _has_index(conn, index_name: str) -> bool:
    return conn.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = :ix)"
        ),
        {"ix": index_name},
    ).scalar()


def _column_is_nullable(conn, table: str, column: str) -> bool:
    row = conn.execute(
        sa.text(
            "SELECT is_nullable FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = :tbl AND column_name = :col"
        ),
        {"tbl": table, "col": column},
    ).scalar()
    return row == "YES"


def _backfill_agent_audit_nulls(conn) -> None:
    """Fill NULL audit FKs before restoring NOT NULL (rows written via M2M API key)."""
    has_null = conn.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM agent WHERE created_by IS NULL OR updated_by IS NULL)"
        )
    ).scalar()
    if not has_null:
        return

    # Prefer a user linked to the agent's tenant; fall back to any user in the DB.
    conn.execute(
        sa.text(
            """
            UPDATE agent AS a
            SET created_by = COALESCE(
                a.created_by,
                (
                    SELECT uta.user_id
                    FROM user_tenant_association uta
                    WHERE uta.tenant_id = a.tenant_id
                    ORDER BY uta.user_id
                    LIMIT 1
                ),
                (SELECT u.id FROM "user" u ORDER BY u.id LIMIT 1)
            )
            WHERE a.created_by IS NULL
            """
        )
    )
    conn.execute(
        sa.text(
            """
            UPDATE agent AS a
            SET updated_by = COALESCE(
                a.updated_by,
                a.created_by,
                (
                    SELECT uta.user_id
                    FROM user_tenant_association uta
                    WHERE uta.tenant_id = a.tenant_id
                    ORDER BY uta.user_id
                    LIMIT 1
                ),
                (SELECT u.id FROM "user" u ORDER BY u.id LIMIT 1)
            )
            WHERE a.updated_by IS NULL
            """
        )
    )

    remaining = conn.execute(
        sa.text(
            "SELECT COUNT(*) FROM agent WHERE created_by IS NULL OR updated_by IS NULL"
        )
    ).scalar()
    if remaining:
        raise RuntimeError(
            f"Cannot restore NOT NULL on agent.created_by/updated_by: "
            f"{remaining} row(s) still have NULL audit user IDs (no users in DB to backfill)."
        )


def _restore_agent_audit_not_null(conn, column: str) -> None:
    if not _has_column(conn, "agent", column):
        return
    if not _column_is_nullable(conn, "agent", column):
        return
    op.alter_column(
        "agent",
        column,
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=False,
    )


def upgrade() -> None:
    conn = op.get_bind()

    if not _has_column(conn, "agent", "status"):
        op.add_column(
            "agent",
            sa.Column(
                "status",
                sa.String(length=20),
                nullable=False,
                server_default="active",
            ),
        )

    if not _has_column(conn, "agent", "llm_model"):
        op.add_column(
            "agent",
            sa.Column("llm_model", sa.String(length=100), nullable=True),
        )

    if not _has_column(conn, "agent", "tts_provider_slug"):
        op.add_column(
            "agent",
            sa.Column("tts_provider_slug", sa.String(length=40), nullable=True),
        )

    if not _has_column(conn, "agent", "tts_voice_external_id"):
        op.add_column(
            "agent",
            sa.Column("tts_voice_external_id", sa.String(length=255), nullable=True),
        )

    if not _has_column(conn, "agent", "tts_language"):
        op.add_column(
            "agent",
            sa.Column("tts_language", sa.String(length=20), nullable=True),
        )

    if not _has_column(conn, "agent", "encrypted_elevenlabs_api_key"):
        op.add_column(
            "agent",
            sa.Column("encrypted_elevenlabs_api_key", sa.Text(), nullable=True),
        )

    if not _has_index(conn, "ix_agent_status"):
        op.create_index(
            "ix_agent_status",
            "agent",
            ["status"],
            unique=False,
        )

    # Relax audit FKs so M2M (API key) requests can create/update.
    op.alter_column(
        "agent",
        "created_by",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=True,
    )
    op.alter_column(
        "agent",
        "updated_by",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=True,
    )


def downgrade() -> None:
    conn = op.get_bind()

    _backfill_agent_audit_nulls(conn)
    _restore_agent_audit_not_null(conn, "updated_by")
    _restore_agent_audit_not_null(conn, "created_by")

    if _has_index(conn, "ix_agent_status"):
        op.drop_index("ix_agent_status", table_name="agent")

    for column in (
        "encrypted_elevenlabs_api_key",
        "tts_language",
        "tts_voice_external_id",
        "tts_provider_slug",
        "llm_model",
        "status",
    ):
        if _has_column(conn, "agent", column):
            op.drop_column("agent", column)
