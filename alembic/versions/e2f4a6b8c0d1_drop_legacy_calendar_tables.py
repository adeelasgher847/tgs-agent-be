"""drop legacy calendar tables

Revision ID: e2f4a6b8c0d1
Revises: d9e4f6a1b2c3
Create Date: 2026-04-02 23:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision: str = "e2f4a6b8c0d1"
down_revision: Union[str, Sequence[str], None] = "d9e4f6a1b2c3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


LEGACY_TABLES = ("business_hours", "blocked_slot")


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    for table_name in LEGACY_TABLES:
        if not inspector.has_table(table_name):
            continue

        row_count = bind.execute(sa.text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar_one()
        if row_count:
            raise RuntimeError(
                f"Refusing to drop legacy table '{table_name}' because it still contains data."
            )

        op.drop_table(table_name)


def downgrade() -> None:
    op.create_table(
        "business_hours",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("day_of_week", sa.Integer(), nullable=False),
        sa.Column("open_time", sa.Time(), nullable=True),
        sa.Column("close_time", sa.Time(), nullable=True),
        sa.Column("is_closed", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("timezone", sa.String(length=60), nullable=False, server_default="UTC"),
        sa.Column("slot_duration_minutes", sa.Integer(), nullable=False, server_default="30"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "blocked_slot",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("blocked_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("blocked_until", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
