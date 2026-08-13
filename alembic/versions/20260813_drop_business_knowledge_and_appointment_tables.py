"""drop business_knowledge and appointment tables

Revision ID: 20260813_drop_bk_appt
Revises: 5b48ce943ea5
Create Date: 2026-08-13 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "20260813_drop_bk_appt"
down_revision: Union[str, Sequence[str], None] = "5b48ce943ea5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Use IF EXISTS so this is safe to run on any DB regardless of
    # whether the table was named business_knowledge (underscore) or
    # businessknowledge (SQLAlchemy auto-lowered name).
    op.execute("DROP TABLE IF EXISTS business_knowledge CASCADE")
    op.execute("DROP TABLE IF EXISTS businessknowledge CASCADE")
    op.execute("DROP TABLE IF EXISTS appointment CASCADE")
    op.execute("DROP INDEX IF EXISTS uq_agent_tenant_follow_up_active")
    op.execute("DROP INDEX IF EXISTS uq_agent_single_follow_up_per_tenant")
    op.execute("ALTER TABLE agent DROP COLUMN IF EXISTS is_follow_up_agent CASCADE")


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())

    if "agent" in inspector.get_table_names():
        column_names = {col["name"] for col in inspector.get_columns("agent")}
        if "is_follow_up_agent" not in column_names:
            op.add_column(
                "agent",
                sa.Column(
                    "is_follow_up_agent",
                    sa.Boolean(),
                    nullable=False,
                    server_default=sa.text("false"),
                ),
            )
        index_names = {ix["name"] for ix in inspector.get_indexes("agent")}
        if "uq_agent_tenant_follow_up_active" not in index_names:
            op.create_index(
                "uq_agent_tenant_follow_up_active",
                "agent",
                ["tenant_id"],
                unique=True,
                postgresql_where=sa.text("is_follow_up_agent = true AND is_deleted = false"),
            )

    if "appointment" not in inspector.get_table_names():
        op.create_table(
            "appointment",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenant.id"), nullable=False, index=True),
            sa.Column("agent_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("agent.id"), nullable=True, index=True),
            sa.Column("call_session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("callsession.id"), nullable=True),
            sa.Column("customer_name", sa.String(length=255), nullable=False),
            sa.Column("customer_phone", sa.String(length=50), nullable=False),
            sa.Column("customer_email", sa.String(length=255), nullable=True),
            sa.Column("appointment_reason", sa.Text(), nullable=True),
            sa.Column("slot_start", sa.DateTime(timezone=True), nullable=False, index=True),
            sa.Column("slot_end", sa.DateTime(timezone=True), nullable=False),
            sa.Column("duration_minutes", sa.Integer(), nullable=False, server_default="30"),
            sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
            sa.Column("created_via", sa.String(length=20), nullable=False, server_default="web"),
            sa.Column("notes", sa.Text(), nullable=True),
            sa.Column("cancellation_reason", sa.Text(), nullable=True),
            sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("reviewed_by_user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("user.id"), nullable=True),
            sa.Column("customer_notified_on_review_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("follow_up_crm_item_id", sa.String(length=128), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index(
            "uq_appointment_tenant_slot_active",
            "appointment",
            ["tenant_id", "slot_start"],
            unique=True,
            postgresql_where=sa.text("status != 'cancelled'"),
        )

    if "business_knowledge" not in inspector.get_table_names():
        op.create_table(
            "business_knowledge",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenant.id"), nullable=False, index=True),
            sa.Column("agent_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("agent.id"), nullable=True, index=True),
            sa.Column("label", sa.String(length=255), nullable=False),
            sa.Column("business_name", sa.String(length=255), nullable=True),
            sa.Column("business_type", sa.String(length=255), nullable=True),
            sa.Column("business_description", sa.Text(), nullable=True),
            sa.Column("address", sa.Text(), nullable=True),
            sa.Column("phone", sa.String(length=255), nullable=True),
            sa.Column("email", sa.String(length=255), nullable=True),
            sa.Column("website_url", sa.String(length=512), nullable=True),
            sa.Column("primary_service", sa.Text(), nullable=True),
            sa.Column("secondary_service", sa.Text(), nullable=True),
            sa.Column("service_areas", sa.Text(), nullable=True),
            sa.Column("specializations", sa.Text(), nullable=True),
            sa.Column("pricing_information", sa.Text(), nullable=True),
            sa.Column("additional_information", sa.Text(), nullable=True),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        )
