"""add appointment review acknowledgement fields

Revision ID: f6b7c8d9e0f1
Revises: e2f4a6b8c0d1
Create Date: 2026-04-10 19:25:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "f6b7c8d9e0f1"
down_revision: Union[str, Sequence[str], None] = "e2f4a6b8c0d1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "appointment",
        sa.Column("review_status", sa.String(length=20), nullable=False, server_default="not_reviewed"),
    )
    op.add_column(
        "appointment",
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "appointment",
        sa.Column("reviewed_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "appointment",
        sa.Column("customer_notified_on_review_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_appointment_reviewed_by_user_id_user",
        "appointment",
        "user",
        ["reviewed_by_user_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint("fk_appointment_reviewed_by_user_id_user", "appointment", type_="foreignkey")
    op.drop_column("appointment", "customer_notified_on_review_at")
    op.drop_column("appointment", "reviewed_by_user_id")
    op.drop_column("appointment", "reviewed_at")
    op.drop_column("appointment", "review_status")
