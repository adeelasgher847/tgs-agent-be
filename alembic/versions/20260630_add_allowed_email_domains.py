"""Add allowed_email_domains to sso_configs

Revision ID: 20260630_add_allowed_email_domains
Revises: 20260626_add_payment_records
Create Date: 2026-06-30
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '20260630_add_allowed_email_domains'
down_revision: Union[str, Sequence[str], None] = '20260626_add_payment_records'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'ssoconfig',
        sa.Column(
            'allowed_email_domains',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            server_default='[]',
        )
    )


def downgrade() -> None:
    op.drop_column('ssoconfig', 'allowed_email_domains')
