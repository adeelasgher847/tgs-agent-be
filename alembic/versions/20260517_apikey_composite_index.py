"""add composite index on apikey (key_hash, tenant_id)

Revision ID: 20260517_apikey_idx
Revises: 20260516_key_prefix
Create Date: 2026-05-17 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

revision: str = "20260517_apikey_idx"
down_revision: Union[str, Sequence[str], None] = "20260516_key_prefix"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Composite index for middleware lookup: WHERE key_hash = ? AND tenant_id = ?
    op.create_index(
        "ix_apikey_key_hash_tenant_id",
        "apikey",
        ["key_hash", "tenant_id"],
        unique=False,
    )
    # Redundant with UNIQUE(key_hash) + composite left-prefix; drop standalone hash index.
    op.drop_index("ix_apikey_key_hash", table_name="apikey")


def downgrade() -> None:
    op.create_index("ix_apikey_key_hash", "apikey", ["key_hash"], unique=False)
    op.drop_index("ix_apikey_key_hash_tenant_id", table_name="apikey")
