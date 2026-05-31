"""stripe checkout fulfillment idempotency table

Revision ID: 20260518_stripe_fulfillment
Revises: 20260518_tenant_name_uq
Create Date: 2026-05-18 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260518_stripe_fulfillment"
down_revision: Union[str, Sequence[str], None] = "20260518_tenant_name_uq"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "stripecheckoutfulfillment",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("checkout_session_id", sa.String(length=255), nullable=False),
        sa.Column("stripe_event_id", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("checkout_session_id"),
    )
    op.create_index(
        op.f("ix_stripecheckoutfulfillment_checkout_session_id"),
        "stripecheckoutfulfillment",
        ["checkout_session_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_stripecheckoutfulfillment_stripe_event_id"),
        "stripecheckoutfulfillment",
        ["stripe_event_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_stripecheckoutfulfillment_stripe_event_id"),
        table_name="stripecheckoutfulfillment",
    )
    op.drop_index(
        op.f("ix_stripecheckoutfulfillment_checkout_session_id"),
        table_name="stripecheckoutfulfillment",
    )
    op.drop_table("stripecheckoutfulfillment")
