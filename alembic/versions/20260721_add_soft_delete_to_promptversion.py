"""add is_deleted to promptversion

Revision ID: 20260721_prompt_soft_del
Revises: 20260716_amd_voicemail
Create Date: 2026-07-21

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260721_prompt_soft_del"
down_revision: Union[str, Sequence[str], None] = "0e45008a4e03"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "promptversion",
        sa.Column("is_deleted", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )


def downgrade() -> None:
    op.drop_column("promptversion", "is_deleted")
