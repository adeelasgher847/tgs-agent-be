"""Add contact_email to Tenant

Revision ID: 7cc13a4806d8
Revises: fdff731d11a7
Create Date: 2026-06-24 18:52:52.448789

"""
import pgvector
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '7cc13a4806d8'
down_revision: Union[str, Sequence[str], None] = 'fdff731d11a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('tenant', sa.Column('contact_email', sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('tenant', 'contact_email')
