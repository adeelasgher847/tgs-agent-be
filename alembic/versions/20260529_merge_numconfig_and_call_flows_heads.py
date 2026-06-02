"""merge numconfig rename and call flows heads

Revision ID: 20260529_merge_heads
Revises: 20260526_numconfig_rename, 20260529_call_flows
Create Date: 2026-05-29

No-op merge: both branches already applied their schema changes independently.
Does not create or alter any tables.
"""
from typing import Sequence, Union

revision: str = "20260529_merge_heads"
down_revision: Union[str, Sequence[str], None] = (
    "20260526_numconfig_rename",
    "20260529_call_flows",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
