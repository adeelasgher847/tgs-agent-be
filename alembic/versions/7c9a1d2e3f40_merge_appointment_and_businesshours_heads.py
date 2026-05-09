"""merge appointment and businesshours heads

Revision ID: 7c9a1d2e3f40
Revises: f6b7c8d9e0f1, ab12cd34ef56
Create Date: 2026-04-10 20:30:00.000000
"""

from typing import Sequence, Union


revision: str = "7c9a1d2e3f40"
down_revision: Union[str, Sequence[str], None] = ("f6b7c8d9e0f1", "ab12cd34ef56")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
