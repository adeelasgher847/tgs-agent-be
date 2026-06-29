"""merge sso and dev heads and rename to ssoconfig

Revision ID: 1def74a4fa8c
Revises: b1c2d3e4f5a6, e3ab5bd291be
Create Date: 2026-06-30 00:59:54.357202

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1def74a4fa8c'
down_revision: Union[str, Sequence[str], None] = ('b1c2d3e4f5a6', 'e3ab5bd291be')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.rename_table('sso_configs', 'ssoconfig')
    op.execute("ALTER INDEX IF EXISTS idx_sso_configs_workspace_id RENAME TO idx_ssoconfig_workspace_id")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("ALTER INDEX IF EXISTS idx_ssoconfig_workspace_id RENAME TO idx_sso_configs_workspace_id")
    op.rename_table('ssoconfig', 'sso_configs')
