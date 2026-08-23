"""merge_billing_and_visual_flow_heads

Revision ID: 57d29c10d96d
Revises: 20260824_rename_compiled_plan, 3f67072326de
Create Date: 2026-08-23 14:53:56.864281

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '57d29c10d96d'
down_revision: Union[str, Sequence[str], None] = ('20260824_rename_compiled_plan', '3f67072326de')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
