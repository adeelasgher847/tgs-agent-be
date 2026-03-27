"""enforce global unique phone numbers

Revision ID: a3a08e39e50b
Revises: 
Create Date: 2026-03-26 19:27:49.982713

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'a3a08e39e50b'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_constraint(op.f('uq_phone_number_per_tenant'), 'phonenumber', type_='unique')
    op.create_unique_constraint('uq_phone_number_global', 'phonenumber', ['phone_number'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint('uq_phone_number_global', 'phonenumber', type_='unique')
    op.create_unique_constraint(op.f('uq_phone_number_per_tenant'), 'phonenumber', ['tenant_id', 'phone_number'], postgresql_nulls_not_distinct=False)
