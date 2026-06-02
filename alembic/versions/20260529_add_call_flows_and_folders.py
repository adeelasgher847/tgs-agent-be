"""add call_flows, prompt_versions, folders, folder_flows, callsession.call_flow_id

Revision ID: 20260529_call_flows
Revises: a1b2c3d4e5f6
Create Date: 2026-05-29
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260529_call_flows"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def _has_table(conn, table: str) -> bool:
    return conn.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = :tbl)"
        ),
        {"tbl": table},
    ).scalar()


def _has_column(conn, table: str, column: str) -> bool:
    return conn.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = :tbl AND column_name = :col)"
        ),
        {"tbl": table, "col": column},
    ).scalar()


def _has_constraint(conn, table: str, constraint: str) -> bool:
    return conn.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.table_constraints "
            "WHERE table_schema = 'public' AND table_name = :tbl AND constraint_name = :con)"
        ),
        {"tbl": table, "con": constraint},
    ).scalar()


def upgrade() -> None:
    conn = op.get_bind()

    # ── callflow ──────────────────────────────────────────────────────────
    if not _has_table(conn, "callflow"):
        op.create_table(
            "callflow",
            sa.Column(
                "id",
                postgresql.UUID(as_uuid=True),
                primary_key=True,
                server_default=sa.text("gen_random_uuid()"),
            ),
            sa.Column(
                "tenant_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("tenant.id"),
                nullable=False,
            ),
            sa.Column(
                "agent_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("agent.id"),
                nullable=False,
            ),
            sa.Column("name", sa.String(255), nullable=False),
            sa.Column("direction", sa.String(20), nullable=False),
            sa.Column("welcome_message_type", sa.String(50), nullable=True),
            sa.Column("custom_welcome_message", sa.Text(), nullable=True),
            # current_prompt_id FK to promptversion added below after that table is created
            sa.Column(
                "current_prompt_id",
                postgresql.UUID(as_uuid=True),
                nullable=True,
            ),
            sa.Column(
                "flow_data",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=True,
            ),
            sa.Column(
                "settings",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=True,
            ),
            sa.Column(
                "is_deleted",
                sa.Boolean(),
                nullable=False,
                server_default="false",
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=True,
            ),
        )
        op.create_index("ix_callflow_tenant_id", "callflow", ["tenant_id"])
        op.create_index("ix_callflow_id", "callflow", ["id"])

    # ── promptversion ─────────────────────────────────────────────────────
    if not _has_table(conn, "promptversion"):
        op.create_table(
            "promptversion",
            sa.Column(
                "id",
                postgresql.UUID(as_uuid=True),
                primary_key=True,
                server_default=sa.text("gen_random_uuid()"),
            ),
            sa.Column(
                "flow_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("callflow.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("prompt_text", sa.Text(), nullable=False),
            sa.Column("gemini_prompt", sa.Text(), nullable=True),
            sa.Column("notes", sa.Text(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=False,
            ),
        )
        op.create_index("ix_promptversion_id", "promptversion", ["id"])
        op.create_index(
            "ix_promptversion_flow_created",
            "promptversion",
            ["flow_id", "created_at"],
        )

    # Now add the deferred FK from callflow.current_prompt_id → promptversion.id
    if not _has_constraint(conn, "callflow", "fk_callflow_current_prompt"):
        op.create_foreign_key(
            "fk_callflow_current_prompt",
            "callflow",
            "promptversion",
            ["current_prompt_id"],
            ["id"],
        )

    # ── folder ────────────────────────────────────────────────────────────
    if not _has_table(conn, "folder"):
        op.create_table(
            "folder",
            sa.Column(
                "id",
                postgresql.UUID(as_uuid=True),
                primary_key=True,
                server_default=sa.text("gen_random_uuid()"),
            ),
            sa.Column(
                "tenant_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("tenant.id"),
                nullable=False,
            ),
            sa.Column("name", sa.String(255), nullable=False),
            sa.Column(
                "is_deleted",
                sa.Boolean(),
                nullable=False,
                server_default="false",
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=True,
            ),
        )
        op.create_index("ix_folder_id", "folder", ["id"])
        op.create_index("ix_folder_tenant_id", "folder", ["tenant_id"])

    # ── folderflow ────────────────────────────────────────────────────────
    if not _has_table(conn, "folderflow"):
        op.create_table(
            "folderflow",
            sa.Column(
                "id",
                postgresql.UUID(as_uuid=True),
                primary_key=True,
                server_default=sa.text("gen_random_uuid()"),
            ),
            sa.Column(
                "folder_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("folder.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "flow_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("callflow.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.UniqueConstraint("folder_id", "flow_id", name="uq_folderflow_folder_flow"),
        )
        op.create_index("ix_folderflow_folder_id", "folderflow", ["folder_id"])
        op.create_index("ix_folderflow_flow_id", "folderflow", ["flow_id"])

    # ── callsession.call_flow_id ──────────────────────────────────────────
    if not _has_column(conn, "callsession", "call_flow_id"):
        op.add_column(
            "callsession",
            sa.Column(
                "call_flow_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("callflow.id"),
                nullable=True,
            ),
        )
        op.create_index("ix_callsession_call_flow_id", "callsession", ["call_flow_id"])


def downgrade() -> None:
    conn = op.get_bind()

    if _has_column(conn, "callsession", "call_flow_id"):
        op.drop_index("ix_callsession_call_flow_id", table_name="callsession")
        op.drop_column("callsession", "call_flow_id")

    if _has_table(conn, "folderflow"):
        op.drop_table("folderflow")

    if _has_table(conn, "folder"):
        op.drop_table("folder")

    if _has_constraint(conn, "callflow", "fk_callflow_current_prompt"):
        op.drop_constraint("fk_callflow_current_prompt", "callflow", type_="foreignkey")

    if _has_table(conn, "promptversion"):
        op.drop_table("promptversion")

    if _has_table(conn, "callflow"):
        op.drop_table("callflow")
