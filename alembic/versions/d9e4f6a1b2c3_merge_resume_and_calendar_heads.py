"""merge resume and calendar heads

Revision ID: d9e4f6a1b2c3
Revises: 20260402_resume_batch_id, c4f1a2b3d5e6
Create Date: 2026-04-02 22:55:00.000000

"""
from typing import Sequence, Union


revision: str = "d9e4f6a1b2c3"
down_revision: Union[str, Sequence[str], None] = ("20260402_resume_batch_id", "c4f1a2b3d5e6")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
