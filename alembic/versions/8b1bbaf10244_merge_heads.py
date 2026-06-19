"""merge heads

Revision ID: 8b1bbaf10244
Revises: 20260618_public_call_token, 56fc0df0339f
Create Date: 2026-06-19 23:13:14.028551

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8b1bbaf10244'
down_revision: Union[str, Sequence[str], None] = ('20260618_public_call_token', '56fc0df0339f')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
