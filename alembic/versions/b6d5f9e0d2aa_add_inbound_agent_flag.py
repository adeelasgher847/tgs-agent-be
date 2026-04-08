"""add inbound agent flag

Revision ID: b6d5f9e0d2aa
Revises: a3a08e39e50b
Create Date: 2026-03-26 20:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b6d5f9e0d2aa"
down_revision: Union[str, Sequence[str], None] = "a3a08e39e50b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent",
        sa.Column(
            "is_inbound_agent",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.create_index(
        "uq_agent_single_inbound_per_tenant",
        "agent",
        ["tenant_id"],
        unique=True,
        postgresql_where=sa.text("is_inbound_agent = true AND is_deleted = false"),
    )


def downgrade() -> None:
    op.drop_index("uq_agent_single_inbound_per_tenant", table_name="agent")
    op.drop_column("agent", "is_inbound_agent")
