"""add calendar tables

Revision ID: c4f1a2b3d5e6
Revises: 20260402_resume_batch_id
Create Date: 2026-04-02 22:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql


revision: str = "c4f1a2b3d5e6"
down_revision: Union[str, Sequence[str], None] = "20260402_resume_batch_id"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_index(inspector, table_name: str, index_name: str) -> bool:
    return any(idx["name"] == index_name for idx in inspector.get_indexes(table_name))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    if not inspector.has_table("businesshours"):
        op.create_table(
            "businesshours",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
            sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenant.id"), nullable=False),
            sa.Column("day_of_week", sa.Integer(), nullable=False),
            sa.Column("open_time", sa.Time(), nullable=True),
            sa.Column("close_time", sa.Time(), nullable=True),
            sa.Column("is_closed", sa.Boolean(), nullable=False, server_default=sa.text("false")),
            sa.Column("timezone", sa.String(length=60), nullable=False, server_default="UTC"),
            sa.Column("slot_duration_minutes", sa.Integer(), nullable=False, server_default="30"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.UniqueConstraint("tenant_id", "day_of_week", name="uq_businesshours_tenant_day"),
        )
    if not _has_index(inspector, "businesshours", "ix_businesshours_tenant_id"):
        op.create_index("ix_businesshours_tenant_id", "businesshours", ["tenant_id"], unique=False)

    if not inspector.has_table("blockedslot"):
        op.create_table(
            "blockedslot",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
            sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenant.id"), nullable=False),
            sa.Column("title", sa.String(length=255), nullable=False),
            sa.Column("blocked_from", sa.DateTime(timezone=True), nullable=False),
            sa.Column("blocked_until", sa.DateTime(timezone=True), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
        )
    if not _has_index(inspector, "blockedslot", "ix_blockedslot_tenant_id"):
        op.create_index("ix_blockedslot_tenant_id", "blockedslot", ["tenant_id"], unique=False)

    if not inspector.has_table("appointment"):
        op.create_table(
            "appointment",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
            sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenant.id"), nullable=False),
            sa.Column("agent_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("agent.id"), nullable=True),
            sa.Column("call_session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("callsession.id"), nullable=True),
            sa.Column("customer_name", sa.String(length=255), nullable=False),
            sa.Column("customer_phone", sa.String(length=50), nullable=False),
            sa.Column("customer_email", sa.String(length=255), nullable=True),
            sa.Column("appointment_reason", sa.Text(), nullable=True),
            sa.Column("slot_start", sa.DateTime(timezone=True), nullable=False),
            sa.Column("slot_end", sa.DateTime(timezone=True), nullable=False),
            sa.Column("duration_minutes", sa.Integer(), nullable=False, server_default="30"),
            sa.Column("status", sa.String(length=20), nullable=False, server_default="confirmed"),
            sa.Column("created_via", sa.String(length=20), nullable=False, server_default="web"),
            sa.Column("notes", sa.Text(), nullable=True),
            sa.Column("cancellation_reason", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        )

    inspector = inspect(bind)
    if not _has_index(inspector, "appointment", "ix_appointment_tenant_id"):
        op.create_index("ix_appointment_tenant_id", "appointment", ["tenant_id"], unique=False)
    if not _has_index(inspector, "appointment", "ix_appointment_agent_id"):
        op.create_index("ix_appointment_agent_id", "appointment", ["agent_id"], unique=False)
    if not _has_index(inspector, "appointment", "ix_appointment_slot_start"):
        op.create_index("ix_appointment_slot_start", "appointment", ["slot_start"], unique=False)

    op.execute("ALTER TABLE appointment DROP CONSTRAINT IF EXISTS uq_appointment_agent_slot")
    op.execute("DROP INDEX IF EXISTS uq_appointment_agent_slot")
    op.execute("DROP INDEX IF EXISTS uq_appointment_tenant_slot_no_agent")
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_appointment_tenant_slot_active
        ON appointment (tenant_id, slot_start)
        WHERE status != 'cancelled'
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_appointment_tenant_slot_active")
    op.execute(
        """
        ALTER TABLE appointment
        ADD CONSTRAINT uq_appointment_agent_slot UNIQUE (agent_id, slot_start)
        """
    )
