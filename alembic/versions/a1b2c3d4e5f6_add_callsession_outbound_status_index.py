"""add_callsession_outbound_status_index

Adds a composite index on callsession(tenant_id, call_type, status) to support
efficient per-workspace concurrent outbound call count queries.

Revision ID: a1b2c3d4e5f6
Revises: f6b7c8d9e0f1
Create Date: 2026-06-08
"""

from typing import Sequence, Union

from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "f6b7c8d9e0f1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_callsession_tenant_calltype_status",
        "callsession",
        ["tenant_id", "call_type", "status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_callsession_tenant_calltype_status", table_name="callsession")
