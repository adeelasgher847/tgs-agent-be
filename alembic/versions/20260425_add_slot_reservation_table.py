"""Add slotreservation for in-call holds until post-call booking.

Revision ID: 20260425_add_slot_reservation
Revises: 20260415_add_tts_provider_and_voice, 20260420_resume_candidate_status
Create Date: 2026-04-25
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql

revision: str = "20260425_add_slot_reservation"
down_revision: Union[str, Sequence[str], None] = (
    "20260415_add_tts_provider_and_voice",
    "20260420_resume_candidate_status",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(inspector, name: str) -> bool:
    return name in inspector.get_table_names()


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if _has_table(inspector, "slotreservation"):
        return

    op.create_table(
        "slotreservation",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenant.id"), nullable=False),
        sa.Column("call_session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("callsession.id"), nullable=False),
        sa.Column("agent_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("agent.id"), nullable=True),
        sa.Column("slot_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("slot_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="active"),
        sa.Column("metadata_json", postgresql.JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_slotreservation_tenant_id", "slotreservation", ["tenant_id"], unique=False)
    op.create_index("ix_slotreservation_call_session_id", "slotreservation", ["call_session_id"], unique=False)
    op.create_index("ix_slotreservation_agent_id", "slotreservation", ["agent_id"], unique=False)
    op.create_index("ix_slotreservation_status", "slotreservation", ["status"], unique=False)
    op.create_index("ix_slotreservation_tenant_active_time", "slotreservation", ["tenant_id", "status", "slot_start"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_slotreservation_tenant_active_time", table_name="slotreservation", if_exists=True)
    op.drop_index("ix_slotreservation_status", table_name="slotreservation", if_exists=True)
    op.drop_index("ix_slotreservation_agent_id", table_name="slotreservation", if_exists=True)
    op.drop_index("ix_slotreservation_call_session_id", table_name="slotreservation", if_exists=True)
    op.drop_index("ix_slotreservation_tenant_id", table_name="slotreservation", if_exists=True)
    op.drop_table("slotreservation")
