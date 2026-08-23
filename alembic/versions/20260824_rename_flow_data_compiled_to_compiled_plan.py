"""Rename callflow.flow_data_compiled to callflow.compiled_plan (BE6-S6-01 Step 4)

Revision ID: 20260824_rename_compiled_plan
Revises: 20260823_gin_flow_data
Create Date: 2026-08-24

Purpose
-------
The ticket's literal schema calls the pre-compiled executor lookup column
``compiled_plan`` ("Store as compiled_plan JSONB column alongside flowData").
The column was originally added as ``flow_data_compiled`` — this migration
renames it in place (no data loss, no new column).
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20260824_rename_compiled_plan"
down_revision: Union[str, Sequence[str], None] = "20260823_gin_flow_data"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column("callflow", "flow_data_compiled", new_column_name="compiled_plan")


def downgrade() -> None:
    op.alter_column("callflow", "compiled_plan", new_column_name="flow_data_compiled")
