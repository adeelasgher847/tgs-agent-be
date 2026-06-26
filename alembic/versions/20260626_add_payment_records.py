"""
Alembic migration: create paymentrecord table for in-call Stripe payments.

Revision ID: 20260626_add_payment_records
Revises: 20260625_add_workspace_slug
Create Date: 2026-06-26
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260626_add_payment_records"
down_revision = "20260625_add_workspace_slug"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "paymentrecord",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "workspace_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenant.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "call_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("callsession.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("payment_intent_id", sa.String(255), nullable=False),
        sa.Column("amount_cents", sa.Integer(), nullable=False),
        sa.Column("currency", sa.String(10), nullable=False, server_default="usd"),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("card_last4", sa.String(4), nullable=True),
        sa.Column("card_brand", sa.String(50), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Primary key index
    op.create_index("ix_paymentrecord_id", "paymentrecord", ["id"])

    # Unique index on Stripe PaymentIntent ID (for idempotent webhook processing)
    op.create_index(
        "uq_paymentrecord_payment_intent_id",
        "paymentrecord",
        ["payment_intent_id"],
        unique=True,
    )

    # Composite index for workspace-scoped listing with time ordering
    op.create_index(
        "idx_paymentrecord_workspace_created",
        "paymentrecord",
        ["workspace_id", "created_at"],
    )

    # Index for call_id FK lookups
    op.create_index("ix_paymentrecord_call_id", "paymentrecord", ["call_id"])
    op.create_index("ix_paymentrecord_workspace_id", "paymentrecord", ["workspace_id"])


def downgrade() -> None:
    op.drop_index("idx_paymentrecord_workspace_created", table_name="paymentrecord")
    op.drop_index("uq_paymentrecord_payment_intent_id", table_name="paymentrecord")
    op.drop_index("ix_paymentrecord_payment_intent_id", table_name="paymentrecord")
    op.drop_index("ix_paymentrecord_call_id", table_name="paymentrecord")
    op.drop_index("ix_paymentrecord_workspace_id", table_name="paymentrecord")
    op.drop_index("ix_paymentrecord_id", table_name="paymentrecord")
    op.drop_table("paymentrecord")
