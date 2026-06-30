"""merge heads 7cc13a4806d8 and f75178a95482

Revision ID: e3ab5bd291be
Revises: 7cc13a4806d8, f75178a95482
Create Date: 2026-06-26 23:32:20.262668

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e3ab5bd291be'
down_revision: Union[str, Sequence[str], None] = ('7cc13a4806d8', 'f75178a95482')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
