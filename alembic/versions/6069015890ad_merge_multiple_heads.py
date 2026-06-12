"""merge multiple heads

Revision ID: 6069015890ad
Revises: 20260611_cleanup_orphan_tables, 20260612_webhook_ssrf_pgcrypto
Create Date: 2026-06-12 21:16:55.167178

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6069015890ad'
down_revision: Union[str, Sequence[str], None] = ('20260611_cleanup_orphan_tables', '20260612_webhook_ssrf_pgcrypto')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
