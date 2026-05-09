"""transferroute table and agent.transfer_route_id

Revision ID: 20260504_transferroute
Revises: 20260430_agent_greeting
Create Date: 2026-05-04 00:00:00.000000

Idempotent: safe if transferroute or agent.transfer_route_id already exists (e.g. manual DDL).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260504_transferroute"
down_revision: Union[str, Sequence[str], None] = "20260430_agent_greeting"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(conn, name: str) -> bool:
    return conn.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = :t)"
        ),
        {"t": name},
    ).scalar()


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


def _has_fk(conn, constraint_name: str) -> bool:
    return conn.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.table_constraints "
            "WHERE constraint_schema = 'public' AND constraint_name = :cn)"
        ),
        {"cn": constraint_name},
    ).scalar()


def upgrade() -> None:
    conn = op.get_bind()

    if not _has_table(conn, "transferroute"):
        op.create_table(
            "transferroute",
            sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("friendly_name", sa.String(length=255), nullable=False),
            sa.Column("phone_number", sa.String(length=20), nullable=False),
            sa.Column("transfer_type", sa.String(length=16), nullable=False, server_default="cold"),
            sa.Column("is_deleted", sa.Boolean(), nullable=False, server_default="false"),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(["tenant_id"], ["tenant.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
    elif not _has_column(conn, "transferroute", "is_deleted"):
        op.add_column(
            "transferroute",
            sa.Column("is_deleted", sa.Boolean(), nullable=False, server_default="false"),
        )

    if not _has_index(conn, "ix_transferroute_tenant_id"):
        op.create_index(
            op.f("ix_transferroute_tenant_id"),
            "transferroute",
            ["tenant_id"],
            unique=False,
        )

    if not _has_column(conn, "agent", "transfer_route_id"):
        op.add_column(
            "agent",
            sa.Column("transfer_route_id", postgresql.UUID(as_uuid=True), nullable=True),
        )
    if not _has_index(conn, "ix_agent_transfer_route_id"):
        op.create_index(
            op.f("ix_agent_transfer_route_id"),
            "agent",
            ["transfer_route_id"],
            unique=False,
        )
    if not _has_fk(conn, "fk_agent_transfer_route_id_transferroute"):
        op.create_foreign_key(
            "fk_agent_transfer_route_id_transferroute",
            "agent",
            "transferroute",
            ["transfer_route_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    conn = op.get_bind()
    if _has_fk(conn, "fk_agent_transfer_route_id_transferroute"):
        op.drop_constraint(
            "fk_agent_transfer_route_id_transferroute", "agent", type_="foreignkey"
        )
    if _has_index(conn, "ix_agent_transfer_route_id"):
        op.drop_index(op.f("ix_agent_transfer_route_id"), table_name="agent")
    if _has_column(conn, "agent", "transfer_route_id"):
        op.drop_column("agent", "transfer_route_id")
    if _has_index(conn, "ix_transferroute_tenant_id"):
        op.drop_index(op.f("ix_transferroute_tenant_id"), table_name="transferroute")
    if _has_table(conn, "transferroute"):
        op.drop_table("transferroute")
