"""Add GIN index on callflow.flow_data for fast JSONB queries (BE6-S6-01 Step 1)

Revision ID: 20260823_gin_flow_data
Revises: 20260817_widen_llm_check_oair
Create Date: 2026-08-23

Purpose
-------
Adds a GIN index on the ``callflow.flow_data`` JSONB column to support fast
containment (``@>``), existence (``?``), and path (``#>``) queries used by the
visual flow editor query paths introduced in Phase 2.

The ``flow_data`` column was added in
``20260529_add_call_flows_and_folders.py``.  No GIN index existed on it (or on
``flow_data_compiled``) prior to this migration.

Transaction note
----------------
This migration runs **inside** an Alembic-managed transaction (the default).
``CREATE INDEX`` without ``CONCURRENTLY`` is therefore used, which is safe and
correct for development and small-to-medium tables.

**On large production tables** the blocking table lock held for the duration of
``CREATE INDEX`` may be unacceptable.  In that case, after applying this
migration you should:

  1. DROP the index created here (``DROP INDEX ix_callflow_flow_data_gin``).
  2. Run ``CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_callflow_flow_data_gin
     ON callflow USING GIN (flow_data);`` **outside** any transaction
     (i.e. directly in psql or via a dedicated script that calls
     ``connection.autocommit = True`` before issuing the DDL).

``CREATE INDEX CONCURRENTLY`` cannot run inside a transaction block, so it
cannot be used inside a standard Alembic migration without non-trivial
workarounds.  Keeping the migration simple and correct is preferred here; the
CONCURRENTLY path is documented above for ops use.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20260823_gin_flow_data"
down_revision: Union[str, Sequence[str], None] = "20260817_widen_llm_check_oair"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_callflow_flow_data_gin "
        "ON callflow USING GIN (flow_data)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_callflow_flow_data_gin")
