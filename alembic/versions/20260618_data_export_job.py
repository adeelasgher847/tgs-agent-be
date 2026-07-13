"""data_export_job

Adds the data_export_job table backing the GDPR data-portability endpoints
(POST/GET /api/v2/workspace/data-export).

Revision ID: 20260618_data_export_job
Revises: 20260618_audit_update_bypass
Create Date: 2026-06-18

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "20260618_data_export_job"
down_revision: Union[str, Sequence[str], None] = "20260618_audit_update_bypass"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "dataexportjob",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("workspace_id", UUID(as_uuid=True), sa.ForeignKey("tenant.id", ondelete="CASCADE"), nullable=False),
        sa.Column("requested_by_user_id", UUID(as_uuid=True), sa.ForeignKey("user.id", ondelete="SET NULL"), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="processing"),
        sa.Column("gcs_path", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_dataexportjob_workspace_id", "dataexportjob", ["workspace_id"])
    op.create_index("ix_dataexportjob_status", "dataexportjob", ["status"])


def downgrade() -> None:
    op.drop_index("ix_dataexportjob_status", table_name="dataexportjob")
    op.drop_index("ix_dataexportjob_workspace_id", table_name="dataexportjob")
    op.drop_table("dataexportjob")
