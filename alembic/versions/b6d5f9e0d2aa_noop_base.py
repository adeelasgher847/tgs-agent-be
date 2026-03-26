"""No-op base migration to satisfy historical DB revision.

Some existing environments are stamped at `b6d5f9e0d2aa`.
This placeholder restores the migration graph so subsequent
tracked migrations can be resolved and executed.
"""

from alembic import op

revision = "b6d5f9e0d2aa"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Intentionally empty: graph placeholder only.
    pass


def downgrade() -> None:
    # Intentionally empty: graph placeholder only.
    pass
