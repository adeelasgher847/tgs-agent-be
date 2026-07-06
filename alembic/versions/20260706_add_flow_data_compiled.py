"""Add flow_data_compiled column to callflow

Revision ID: 20260706_flow_data_compiled
Revises: 20260703_caller_memory
Create Date: 2026-07-06 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "20260706_flow_data_compiled"
down_revision: Union[str, Sequence[str], None] = "20260703_caller_memory"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "callflow",
        sa.Column("flow_data_compiled", JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("callflow", "flow_data_compiled")
