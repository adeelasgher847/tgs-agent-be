"""add batch_jobs and batch_call_records tables

Revision ID: 20260609_batch_calls
Revises: 20260610_outbound_idx
Create Date: 2026-06-09 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260609_batch_calls"
down_revision: Union[str, Sequence[str], None] = "20260610_outbound_idx"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "batchjob",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenant.id", ondelete="CASCADE"), nullable=False),
        sa.Column("agent_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("agent.id", ondelete="SET NULL"), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("total_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("waiting_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("active_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completed_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("gcs_path", sa.Text(), nullable=True),
        sa.Column("scheduled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_batchjob_workspace_id", "batchjob", ["workspace_id"])
    op.create_index("ix_batchjob_agent_id", "batchjob", ["agent_id"])
    op.create_index("ix_batchjob_status", "batchjob", ["status"])

    op.create_table(
        "batchcallrecord",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("batch_job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("batchjob.id", ondelete="CASCADE"), nullable=False),
        sa.Column("phone_number", sa.String(length=50), nullable=False),
        sa.Column("variables", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="waiting"),
        sa.Column("call_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("callsession.id", ondelete="SET NULL"), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_batchcallrecord_batch_job_id", "batchcallrecord", ["batch_job_id"])
    op.create_index("ix_batchcallrecord_status", "batchcallrecord", ["status"])
    op.create_index("ix_batchcallrecord_call_id", "batchcallrecord", ["call_id"])
    # Composite index for the SKIP LOCKED pickup query
    op.create_index(
        "ix_batchcallrecord_job_status_pickup",
        "batchcallrecord",
        ["batch_job_id", "status", "next_attempt_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_batchcallrecord_job_status_pickup", table_name="batchcallrecord")
    op.drop_index("ix_batchcallrecord_call_id", table_name="batchcallrecord")
    op.drop_index("ix_batchcallrecord_status", table_name="batchcallrecord")
    op.drop_index("ix_batchcallrecord_batch_job_id", table_name="batchcallrecord")
    op.drop_table("batchcallrecord")

    op.drop_index("ix_batchjob_status", table_name="batchjob")
    op.drop_index("ix_batchjob_agent_id", table_name="batchjob")
    op.drop_index("ix_batchjob_workspace_id", table_name="batchjob")
    op.drop_table("batchjob")
