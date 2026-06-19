"""remove role constraint and drop rbac_roles

Revision ID: 56fc0df0339f
Revises: 1f194bb62ea0
Create Date: 2026-06-19 21:59:45.850273

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '56fc0df0339f'
down_revision: Union[str, Sequence[str], None] = '20260617_enterprise_audit'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute('ALTER TABLE role DROP CONSTRAINT IF EXISTS chk_role_role')
    op.execute('DROP TABLE IF EXISTS rbac_roles CASCADE')


def downgrade() -> None:
    """Downgrade schema."""
    pass
