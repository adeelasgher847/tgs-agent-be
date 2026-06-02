"""merge stripe fulfillment and schema v1 heads

Revision ID: 20260521_merge_heads
Revises: 20260518_stripe_fulfillment, 20260521_role_config_readonly
Create Date: 2026-05-21 20:00:00.000000

"""
from typing import Sequence, Union

revision: str = "20260521_merge_heads"
down_revision: Union[str, Sequence[str], None] = (
    "20260518_stripe_fulfillment",
    "20260521_role_config_readonly",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
