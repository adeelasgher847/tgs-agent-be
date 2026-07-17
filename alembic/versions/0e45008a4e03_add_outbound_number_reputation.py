"""add_outbound_number_reputation

Revision ID: 0e45008a4e03
Revises: 20260716_amd_voicemail
Create Date: 2026-07-16 23:45:57.383860

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '0e45008a4e03'
down_revision: Union[str, Sequence[str], None] = '20260716_amd_voicemail'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'phonenumberreputation',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('phone_number_id', sa.UUID(), nullable=False),
        sa.Column('reputation_score', sa.Integer(), server_default='100', nullable=False),
        sa.Column('spam_flagged', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('last_checked_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('checked_by', sa.String(), nullable=True),
        sa.Column('flagged_reason', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['phone_number_id'], ['phonenumber.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_phonenumberreputation_phone_number_id'),
        'phonenumberreputation',
        ['phone_number_id'],
        unique=True,
    )
    op.add_column('batchjob', sa.Column('actual_from_number', sa.String(length=20), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('batchjob', 'actual_from_number')
    op.drop_index(op.f('ix_phonenumberreputation_phone_number_id'), table_name='phonenumberreputation')
    op.drop_table('phonenumberreputation')
