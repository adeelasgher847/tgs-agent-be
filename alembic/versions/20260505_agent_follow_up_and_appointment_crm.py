"""agent is_follow_up_agent + appointment follow_up_crm_item_id

Revision ID: 20260505_followup_agent
Revises: 20260504_transferroute
Create Date: 2026-05-05 00:00:00.000000

Note: Apply in staging first; production migration per ops window (see product plan).
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "20260505_followup_agent"
down_revision: Union[str, Sequence[str], None] = "20260504_transferroute"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent",
        sa.Column(
            "is_follow_up_agent",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.create_index(
        "uq_agent_single_follow_up_per_tenant",
        "agent",
        ["tenant_id"],
        unique=True,
        postgresql_where=sa.text("is_follow_up_agent = true AND is_deleted = false"),
    )
    op.add_column(
        "appointment",
        sa.Column("follow_up_crm_item_id", sa.String(length=128), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("appointment", "follow_up_crm_item_id")
    op.drop_index("uq_agent_single_follow_up_per_tenant", table_name="agent")
    op.drop_column("agent", "is_follow_up_agent")
