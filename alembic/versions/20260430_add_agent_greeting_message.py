"""add greeting_message to agent

Revision ID: 20260430_agent_greeting
Revises: 20260429_add_businessknowledge
Create Date: 2026-04-30 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260430_agent_greeting"
down_revision: Union[str, Sequence[str], None] = "20260429_add_businessknowledge"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent",
        sa.Column("greeting_message", sa.Text, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("agent", "greeting_message")
