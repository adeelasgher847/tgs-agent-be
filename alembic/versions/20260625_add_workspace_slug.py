"""Add workspace_slug to Tenant

Revision ID: a9f2b3c4d5e6
Revises: 7cc13a4806d8
Create Date: 2026-06-25 20:30:00.000000

"""
import re
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision: str = 'a9f2b3c4d5e6'
down_revision: Union[str, Sequence[str], None] = '7cc13a4806d8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _slugify(name: str) -> str:
    """URL-safe lowercase slug from a workspace name."""
    slug = name.lower().strip()
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    return slug.strip('-') or 'workspace'


def upgrade() -> None:
    """Add workspace_slug column, backfill existing rows, then add unique index."""
    # Step 1: add nullable column
    op.add_column('tenant', sa.Column('workspace_slug', sa.String(length=100), nullable=True))

    # Step 2: backfill existing rows using a server-side expression
    # We use lower + regexp_replace for Postgres; for SQLite tests we skip (test fixtures
    # set workspace_slug directly when needed).
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == 'postgresql':
        # Postgres: slugify name in SQL, then resolve duplicates in Python
        rows = bind.execute(text("SELECT id, name FROM tenant ORDER BY created_at")).fetchall()
        used: dict[str, int] = {}
        for row_id, name in rows:
            base = re.sub(r'[^a-z0-9]+', '-', (name or 'workspace').lower()).strip('-') or 'workspace'
            slug = base
            if slug in used:
                used[slug] += 1
                slug = f"{base}-{used[slug]}"
            else:
                used[slug] = 0
            bind.execute(
                text("UPDATE tenant SET workspace_slug = :slug WHERE id = :id"),
                {"slug": slug, "id": row_id},
            )

    # Step 3: unique index (partial — only among non-deleted rows)
    op.create_index(
        'uq_tenant_workspace_slug_active',
        'tenant',
        ['workspace_slug'],
        unique=True,
        postgresql_where=sa.text('deleted_at IS NULL'),
        sqlite_where=sa.text('deleted_at IS NULL'),
    )


def downgrade() -> None:
    op.drop_index('uq_tenant_workspace_slug_active', table_name='tenant')
    op.drop_column('tenant', 'workspace_slug')
