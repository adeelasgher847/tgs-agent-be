"""add inbound CRM tables (tenant call log sync)

Revision ID: f1a2b3c4d5e6
Revises: e2f4a6b8c0d1
Create Date: 2026-04-06 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, Sequence[str], None] = "e2f4a6b8c0d1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "tenantinboundcrmconfig",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenant.id"), nullable=False),
        sa.Column("provider", sa.String(length=20), nullable=False, server_default="trello"),
        sa.Column("connection_type", sa.String(length=30), nullable=False, server_default="byo_credentials"),
        sa.Column("encrypted_api_key", sa.Text(), nullable=True),
        sa.Column("encrypted_api_token", sa.Text(), nullable=True),
        sa.Column("container_id", sa.String(length=200), nullable=True),
        sa.Column("container_url", sa.String(length=500), nullable=True),
        sa.Column("default_list_id", sa.String(length=200), nullable=True),
        sa.Column("extra_config", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("is_enabled", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_by_user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("user.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("tenant_id", name="uq_tenantinboundcrmconfig_tenant_id"),
    )
    op.create_index("ix_tenantinboundcrmconfig_container_id", "tenantinboundcrmconfig", ["container_id"], unique=False)

    op.create_table(
        "calllogcrmsync",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("call_log_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("calllog.id"), nullable=False),
        sa.Column(
            "tenant_inbound_crm_config_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenantinboundcrmconfig.id"),
            nullable=False,
        ),
        sa.Column("external_item_id", sa.String(length=200), nullable=True),
        sa.Column("external_item_url", sa.String(length=800), nullable=True),
        sa.Column("sync_status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("call_log_id", name="uq_calllogcrmsync_call_log_id"),
    )
    op.create_index(
        "ix_calllogcrmsync_tenant_inbound_crm_config_id",
        "calllogcrmsync",
        ["tenant_inbound_crm_config_id"],
        unique=False,
    )
    op.create_index("ix_calllogcrmsync_external_item_id", "calllogcrmsync", ["external_item_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_calllogcrmsync_external_item_id", table_name="calllogcrmsync")
    op.drop_index("ix_calllogcrmsync_tenant_inbound_crm_config_id", table_name="calllogcrmsync")
    op.drop_table("calllogcrmsync")

    op.drop_index("ix_tenantinboundcrmconfig_container_id", table_name="tenantinboundcrmconfig")
    op.drop_table("tenantinboundcrmconfig")
