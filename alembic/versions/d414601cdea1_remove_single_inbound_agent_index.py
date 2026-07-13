"""remove_single_inbound_agent_index

Revision ID: d414601cdea1
Revises: 20260706_flow_data_compiled
Create Date: 2026-07-09 00:22:04.433957

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd414601cdea1'
down_revision: Union[str, Sequence[str], None] = '20260706_flow_data_compiled'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_index("uq_agent_single_inbound_per_tenant", table_name="agent")


def downgrade() -> None:
    """Downgrade schema."""
    op.create_index(
        "uq_agent_single_inbound_per_tenant",
        "agent",
        ["tenant_id"],
        unique=True,
        postgresql_where=sa.text("is_inbound_agent = true AND is_deleted = false"),
    )
