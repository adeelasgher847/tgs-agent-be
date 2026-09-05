"""add_sysadmin_portal_tables

Revision ID: a02429359c57
Revises: 86f5457724a9
Create Date: 2026-09-05 00:00:00.000000

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "a02429359c57"
down_revision = "86f5457724a9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Table names follow the repo convention: cls.__name__.lower()
    # SysAdminUser → sysadminuser, SysAdminApiKey → sysadminapikey,
    # SysRequestLog → sysrequestlog, SysRequestStats → sysrequeststats,
    # SysAuditLog → sysauditlog

    op.create_table(
        "sysadminuser",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text("gen_random_uuid()")),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("role", sa.String(50), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_sysadminuser_email", "sysadminuser", ["email"], unique=True)

    op.create_table(
        "sysadminapikey",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text("gen_random_uuid()")),
        sa.Column("admin_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("sysadminuser.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("key_hash", sa.String(255), unique=True, nullable=False),
        sa.Column("key_prefix", sa.String(8), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_sysadminapikey_admin_id", "sysadminapikey", ["admin_id"])

    op.create_table(
        "sysrequestlog",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenant.id", ondelete="SET NULL"), nullable=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("path", sa.String(500), nullable=False),
        sa.Column("method", sa.String(10), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(20), nullable=False, server_default="backend"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("stack_trace", sa.Text(), nullable=True),
        sa.Column("request_id", sa.String(64), nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("country", sa.String(2), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_sysrequestlog_tenant_created", "sysrequestlog", ["tenant_id", "created_at"])
    op.create_index("ix_sysrequestlog_status_created", "sysrequestlog", ["status_code", "created_at"])
    op.create_index("ix_sysrequestlog_path_method", "sysrequestlog", ["path", "method"])

    op.create_table(
        "sysrequeststats",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text("gen_random_uuid()")),
        sa.Column("month", sa.String(7), nullable=False),
        sa.Column("path", sa.String(500), nullable=False),
        sa.Column("method", sa.String(10), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("total_requests", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("success_count", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("error_count", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("total_duration_ms", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("p95_duration_ms", sa.Integer(), nullable=True),
        sa.Column("avg_duration_ms", sa.Integer(), nullable=True),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index(
        "uq_sysrequeststats_month_path_method_tenant",
        "sysrequeststats",
        ["month", "path", "method", "tenant_id"],
        unique=True,
    )

    op.create_table(
        "sysauditlog",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text("gen_random_uuid()")),
        sa.Column("admin_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("sysadminuser.id", ondelete="SET NULL"), nullable=True),
        sa.Column("admin_email", sa.String(255), nullable=False),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("source_page", sa.String(100), nullable=True),
        sa.Column("tenant_schema", sa.String(100), nullable=True),
        sa.Column("details", postgresql.JSONB(), nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_sysauditlog_admin_id", "sysauditlog", ["admin_id"])
    op.create_index("ix_sysauditlog_action", "sysauditlog", ["action"])
    op.create_index("ix_sysauditlog_created_at", "sysauditlog", ["created_at"])


def downgrade() -> None:
    op.drop_table("sysauditlog")
    op.drop_table("sysrequeststats")
    op.drop_table("sysrequestlog")
    op.drop_table("sysadminapikey")
    op.drop_table("sysadminuser")
