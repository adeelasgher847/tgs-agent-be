"""add_role_id_to_invite

Revision ID: fdff731d11a7
Revises: 9f3a2c7e5d41
Create Date: 2026-06-23 19:32:47.749184

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fdff731d11a7'
down_revision: Union[str, Sequence[str], None] = '9f3a2c7e5d41'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'invite',
        sa.Column('role_id', sa.dialects.postgresql.UUID(as_uuid=True), sa.ForeignKey('role.id'), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('invite', 'role_id')

