"""phone provisioning sprint2: provider col, numberconfiguration, agent pending/ready

Revision ID: a1b2c3d4e5f6
Revises: 20260522_agent_ticket
Create Date: 2026-05-26

Changes:
- phonenumber: provider, sip_username, sip_password
- numberconfiguration table (Base tablename convention)
- agent.status pending/ready are app-enforced (no DDL enum)
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "a1b2c3d4e5f6"
down_revision = "20260522_agent_ticket"
branch_labels = None
depends_on = None


def _has_column(conn, table: str, column: str) -> bool:
    return conn.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = :tbl AND column_name = :col)"
        ),
        {"tbl": table, "col": column},
    ).scalar()


def _has_table(conn, table: str) -> bool:
    return conn.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = :tbl)"
        ),
        {"tbl": table},
    ).scalar()


def upgrade() -> None:
    conn = op.get_bind()

    if not _has_column(conn, "phonenumber", "provider"):
        op.add_column(
            "phonenumber",
            sa.Column(
                "provider",
                sa.String(20),
                nullable=False,
                server_default="twilio",
            ),
        )

    if not _has_column(conn, "phonenumber", "sip_username"):
        op.add_column(
            "phonenumber",
            sa.Column("sip_username", sa.String(255), nullable=True),
        )

    if not _has_column(conn, "phonenumber", "sip_password"):
        op.add_column(
            "phonenumber",
            sa.Column("sip_password", sa.Text(), nullable=True),
        )

    if not _has_table(conn, "numberconfiguration"):
        op.create_table(
            "numberconfiguration",
            sa.Column(
                "id",
                postgresql.UUID(as_uuid=True),
                primary_key=True,
                server_default=sa.text("gen_random_uuid()"),
            ),
            sa.Column(
                "phone_number_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("phonenumber.id", ondelete="CASCADE"),
                nullable=False,
                unique=True,
                index=True,
            ),
            sa.Column(
                "recording_enabled",
                sa.Boolean(),
                nullable=False,
                server_default="false",
            ),
            sa.Column(
                "max_duration_seconds",
                sa.Integer(),
                nullable=False,
                server_default="3600",
            ),
            sa.Column(
                "business_hours",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=True,
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
                onupdate=sa.text("now()"),
                nullable=True,
            ),
        )


def downgrade() -> None:
    conn = op.get_bind()
    if _has_table(conn, "numberconfiguration"):
        op.drop_table("numberconfiguration")
    if _has_column(conn, "phonenumber", "sip_password"):
        op.drop_column("phonenumber", "sip_password")
    if _has_column(conn, "phonenumber", "sip_username"):
        op.drop_column("phonenumber", "sip_username")
    if _has_column(conn, "phonenumber", "provider"):
        op.drop_column("phonenumber", "provider")
