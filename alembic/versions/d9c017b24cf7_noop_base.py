"""No-op migration to satisfy existing DB alembic revision.

Your database currently references revision `d9c017b24cf7` in `alembic_version`,
but the corresponding migration file is not present in this repo.

This stub keeps Alembic migration graph consistent so we can apply the new
KB table migration.
"""

from alembic import op

revision = "d9c017b24cf7"
down_revision = "b6d5f9e0d2aa"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Intentionally empty: this migration is a graph placeholder.
    pass


def downgrade() -> None:
    # Intentionally empty: this migration is a graph placeholder.
    pass

