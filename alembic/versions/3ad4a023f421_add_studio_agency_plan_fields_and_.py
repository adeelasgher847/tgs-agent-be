"""add_studio_agency_plan_fields_and_tenant_scoped_subscriptions

Revision ID: 3ad4a023f421
Revises: c4e70c96cb81
Create Date: 2026-08-22 11:05:32.525502

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '3ad4a023f421'
down_revision: Union[str, Sequence[str], None] = 'c4e70c96cb81'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Studio/Agency core-product plan fields on `plan` — all nullable, so
    # existing CRM-addon plan rows (monday/clickup/jira/trello) are unaffected.
    op.add_column('plan', sa.Column('included_minutes', sa.Integer(), nullable=True))
    op.add_column('plan', sa.Column('monthly_credits', sa.Numeric(precision=10, scale=2), nullable=True))
    op.add_column('plan', sa.Column('max_subaccounts', sa.Integer(), nullable=True))
    op.add_column('plan', sa.Column('free_phone_numbers', sa.Integer(), nullable=True))
    op.add_column('plan', sa.Column('features', postgresql.JSONB(astext_type=sa.Text()), nullable=True))

    # Tenant-scoped (core-product) subscriptions on `subscription`, coexisting
    # with the existing user_id + crm_type keyed CRM-addon subscriptions.
    op.add_column('subscription', sa.Column('tenant_id', sa.UUID(), nullable=True))
    op.create_index(op.f('ix_subscription_tenant_id'), 'subscription', ['tenant_id'], unique=False)
    op.create_foreign_key(
        'subscription_tenant_id_fkey', 'subscription', 'tenant', ['tenant_id'], ['id']
    )
    # Partial unique index: at most one active core (non-CRM) subscription per
    # tenant. Legacy/future CRM-addon rows keep tenant_id NULL and are excluded
    # by the predicate — a plain UniqueConstraint on tenant_id wouldn't work
    # here since Postgres doesn't dedupe NULLs across the mixed rows.
    op.create_index(
        'uq_tenant_core_subscription',
        'subscription',
        ['tenant_id'],
        unique=True,
        postgresql_where=sa.text('crm_type IS NULL AND tenant_id IS NOT NULL'),
    )

    # Per-usage-row credit deduction (0 while inside a plan's free-minute
    # allowance, >0 during overage billing). Backfilled to 0 for existing rows
    # via server_default so this is a non-locking additive change.
    op.add_column(
        'usagerecord',
        sa.Column('credits_charged', sa.Numeric(precision=10, scale=4), server_default='0', nullable=False),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('usagerecord', 'credits_charged')

    op.drop_index('uq_tenant_core_subscription', table_name='subscription', postgresql_where=sa.text('crm_type IS NULL AND tenant_id IS NOT NULL'))
    op.drop_constraint('subscription_tenant_id_fkey', 'subscription', type_='foreignkey')
    op.drop_index(op.f('ix_subscription_tenant_id'), table_name='subscription')
    op.drop_column('subscription', 'tenant_id')

    op.drop_column('plan', 'features')
    op.drop_column('plan', 'free_phone_numbers')
    op.drop_column('plan', 'max_subaccounts')
    op.drop_column('plan', 'monthly_credits')
    op.drop_column('plan', 'included_minutes')
