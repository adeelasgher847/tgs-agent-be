"""add calendlyintegration table

Revision ID: b2d3c4e5f6a8
Revises: a1c2b3d4e5f7
Create Date: 2026-07-15 00:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "b2d3c4e5f6a8"
down_revision: Union[str, Sequence[str], None] = "a1c2b3d4e5f7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "calendlyintegration",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenant.id"), nullable=False, index=True),
        sa.Column("access_token", sa.Text(), nullable=True),
        sa.Column("refresh_token", sa.Text(), nullable=True),
        sa.Column("token_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("calendly_user_uri", sa.Text(), nullable=True),
        sa.Column("calendly_event_type_uri", sa.Text(), nullable=True),
        sa.Column("connected_by_user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("user.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("workspace_id", name="uq_calendly_integration_workspace"),
    )


def downgrade() -> None:
    op.drop_table("calendlyintegration")
