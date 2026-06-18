"""user_email_partial_unique

GDPR erasure anonymizes every deleted user's email to the same fixed
placeholder ('[DELETED@DELETED.COM]'). A plain global UNIQUE constraint on
user.email would reject the second such row in the same transaction,
making the literal acceptance criteria of the erasure ticket impossible to
satisfy. Replaces the global unique index with one scoped to active
(deleted_at IS NULL) rows only — mirrors the existing uq_tenant_name_active
pattern on tenant.name.

Revision ID: 20260618_user_email_partial_unique
Revises: 20260618_data_export_job
Create Date: 2026-06-18

"""
from typing import Sequence, Union

from alembic import op

revision: str = "20260618_user_email_partial_unique"
down_revision: Union[str, Sequence[str], None] = "20260618_data_export_job"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # The original unique index was created by SQLAlchemy as
    # Column(unique=True, index=True) -> a single unique index named
    # ix_user_email. Some environments may instead carry a plain UNIQUE
    # table constraint (user_email_key) from a manually-applied schema —
    # both drops are no-ops via IF EXISTS when the object isn't present.
    op.execute('DROP INDEX IF EXISTS ix_user_email')
    op.execute('ALTER TABLE "user" DROP CONSTRAINT IF EXISTS user_email_key')

    op.execute('CREATE INDEX ix_user_email ON "user" (email)')
    op.execute(
        'CREATE UNIQUE INDEX uq_user_email_active ON "user" (email) '
        "WHERE deleted_at IS NULL"
    )


def downgrade() -> None:
    op.execute('DROP INDEX IF EXISTS uq_user_email_active')
    op.execute('DROP INDEX IF EXISTS ix_user_email')
    op.execute('CREATE UNIQUE INDEX ix_user_email ON "user" (email)')
