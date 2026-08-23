"""add_wallet_auto_recharge_trigger_trail

Revision ID: 3f67072326de
Revises: 7ec8c07644d0
Create Date: 2026-08-23 11:20:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "3f67072326de"
down_revision: Union[str, Sequence[str], None] = "7ec8c07644d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        "UPDATE walletautorechargeconfig SET updated_at = created_at WHERE updated_at IS NULL"
    )
    op.alter_column(
        "walletautorechargeconfig",
        "updated_at",
        existing_type=sa.DateTime(timezone=True),
        nullable=False,
    )
    op.add_column(
        "walletautorechargeconfig",
        sa.Column("last_payment_intent_id", sa.String(), nullable=True),
    )
    op.add_column(
        "walletautorechargeconfig",
        sa.Column("last_trigger_status", sa.String(), nullable=True),
    )
    op.add_column(
        "walletautorechargeconfig",
        sa.Column("last_trigger_error", sa.String(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("walletautorechargeconfig", "last_trigger_error")
    op.drop_column("walletautorechargeconfig", "last_trigger_status")
    op.drop_column("walletautorechargeconfig", "last_payment_intent_id")
    op.alter_column(
        "walletautorechargeconfig",
        "updated_at",
        existing_type=sa.DateTime(timezone=True),
        nullable=True,
    )
