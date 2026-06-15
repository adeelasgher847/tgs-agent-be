"""add composite analytics index on callsession

Revision ID: 20260616_analytics_index
Revises: 20260615_callback
Create Date: 2026-06-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260616_analytics_index"
down_revision: Union[str, None] = "20260615_workspace_settings"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Composite index for date-range analytics queries filtered by tenant.
    # Covers GET /calls/history, /calls/history/metrics, /calls/history/time-series.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_callsession_tenant_start_time "
        "ON callsession (tenant_id, start_time DESC)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_callsession_tenant_start_time")
