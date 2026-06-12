"""webhook secret encryption migrated to pgcrypto

Revision ID: 20260612_webhook_ssrf_pgcrypto
Revises: 20260610_webhooks
Create Date: 2026-06-12 00:00:00.000000

Changes
-------
1. Documents the encryption-format transition on webhookendpoint.secret:
   - Secrets written before this revision are JWT-encrypted (legacy).
   - Secrets written from this revision onwards are pgp_sym_encrypt base64.
   - Reads transparently handle both formats via decrypt_stored_webhook_secret().
   - No column type change is required (TEXT accommodates both formats).

2. Adds a PostgreSQL column comment so the format is self-documenting in the
   schema itself.

Operator checklist
------------------
* Set WEBHOOK_SECRET_ENCRYPTION_KEY in environment / Secret Manager before
  deploying — new secrets will fail to write without it.
* Existing JWT-encrypted rows continue to work for reads as long as SECRET_KEY
  remains in the environment.  A background data migration (re-encrypting old
  rows via pgcrypto) can be run separately once the new key is deployed.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "20260612_webhook_ssrf_pgcrypto"
down_revision: Union[str, Sequence[str], None] = "20260610_webhooks"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        COMMENT ON COLUMN webhookendpoint.secret IS
        'HMAC signing secret encrypted at rest. '
        'Format: pgp_sym_encrypt base64 (>= v20260612) or legacy JWT (< v20260612). '
        'Decrypted via decrypt_stored_webhook_secret() in app/core/db_encryption.py. '
        'Never returned to API callers.';
        """
    )


def downgrade() -> None:
    op.execute("COMMENT ON COLUMN webhookendpoint.secret IS NULL;")
