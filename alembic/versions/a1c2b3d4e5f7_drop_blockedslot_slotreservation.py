"""drop blockedslot and slotreservation tables (Calendly migration)

Availability, conflict-checking, and in-call slot holds now live in Calendly.
The `businesshours` table is intentionally NOT dropped here — the Smart
Callback Scheduler (app/services/callback_scheduler_service.py) still reads
it to gate retry timing, unrelated to appointment-slot availability.

Revision ID: a1c2b3d4e5f7
Revises: 20260714_rename_gcs_columns_to_s3
Create Date: 2026-07-15 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "a1c2b3d4e5f7"
down_revision: Union[str, Sequence[str], None] = "20260714_rename_gcs_columns_to_s3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_table("slotreservation")
    op.drop_table("blockedslot")


def downgrade() -> None:
    op.create_table(
        "blockedslot",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenant.id"), nullable=False, index=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("blocked_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("blocked_until", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "slotreservation",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenant.id"), nullable=False, index=True),
        sa.Column("call_session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("callsession.id"), nullable=False, index=True),
        sa.Column("agent_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("agent.id"), nullable=True, index=True),
        sa.Column("slot_start", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("slot_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="active"),
        sa.Column("metadata_json", postgresql.JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_slotreservation_tenant_active_time",
        "slotreservation",
        ["tenant_id", "status", "slot_start"],
    )
