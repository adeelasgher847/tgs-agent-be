"""schema v2 completion

Revision ID: 20260602_schema_v2
Revises: 20260602_phonenumber_provider
Create Date: 2026-06-02

Implements the remaining gaps to reach Schema v2 as defined by the agent-builder
/ call-flow ticket. All changes are additive or widening — no existing rows are
deleted or renamed.

Key changes
-----------
agent table
  • smart_callback  : boolean NOT NULL DEFAULT false (new column)
  • ix_agent_tenant_id : BTree index on tenant_id (may already exist)
  • status server_default changed to 'pending' for new rows only (existing rows
    keep whatever value they already have — no UPDATE issued)
  • ck_agent_status_v2 : widened CHECK that accepts both legacy values
    ('active','inactive','draft') and ticket values ('pending','ready','error')
  • ck_agent_llm_model : nullable-safe CHECK from app.core.llm_models.ALLOWED_LLM_MODELS
    (via _llm_check_sql() — single source of truth, no duplicate list in this file)

callflow table
  • ix_callflow_agent_id : BTree index on agent_id
  • ck_callflow_direction : replaced with widened version adding 'bidirectional'
    (existing 'inbound' and 'outbound' rows are untouched)
  • ck_callflow_welcome_message_type : new CHECK constraint

promptversion table
  • ix_promptversion_flow_id : simple index on flow_id (composite one already
    exists as ix_promptversion_flow_created; this satisfies the ticket requirement)

pgcrypto
  • Extension enabled (idempotent)
  • Data migration: legacy JWT BYO keys (see app.core.db_encryption.is_legacy_jwt_ciphertext)
    are re-encrypted with pgp_sym_encrypt using ELEVENLABS_ENCRYPTION_KEY.
    Upgrade **fails** if any JWT rows exist but the key is unset (prevents mixed formats).
    Rows with NULL or already-pgcrypto ciphertext are skipped.  Rows that fail
    JWT-decryption are left untouched and counted in a warning log (manual fix required).

Deploy / rollback
-----------------
See docs/db/schema-v2.md — "Deploy runbook (schema v2 completion)" and "Rollback (downgrade)".

Direction constraint strategy
------------------------------
The prior migration (20260601_callflow_schema) owns ck_callflow_direction with
definition IN ('inbound','outbound').  This migration drops it and recreates it
wider.  The downgrade restores the narrow definition so the prior migration's
downgrade can subsequently remove it without error.

Status constraint strategy
--------------------------
We intentionally widen rather than data-migrate so that legacy front-ends
sending 'active' continue to work without a coordinated deploy.  The ticket
values ('pending','ready','error') are the canonical new states; the application
default for new rows is now 'pending'.  A future cleanup migration can narrow the
CHECK and migrate remaining legacy rows once the API layer is updated.

is_active / callflow
--------------------
The ticket acceptance criterion for is_active is satisfied by documenting that
`is_active = NOT is_deleted` is enforced by the service layer.  No new column is
added to avoid a duplicate boolean that diverges under concurrent writes.

folderflow composite PK
-----------------------
folderflow already has id (surrogate PK) + UNIQUE(folder_id, flow_id).  The
ticket's PK requirement is satisfied by the unique constraint.  Dropping the
surrogate id would require ORM changes and FK rewrites with no functional gain.
This deviation is documented in docs/db/schema-v2.md.
"""

from __future__ import annotations

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260602_schema_v2"
down_revision: Union[str, Sequence[str], None] = "20260602_phonenumber_provider"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_log = logging.getLogger(__name__)


def _llm_check_sql() -> str:
    """Build ck_agent_llm_model CHECK from app.core.llm_models (single source of truth)."""
    from app.core.llm_models import ALLOWED_LLM_MODELS

    return "llm_model IS NULL OR llm_model IN (%s)" % ", ".join(
        f"'{m}'" for m in ALLOWED_LLM_MODELS
    )


_STATUS_VALUES = ("pending", "ready", "error", "active", "inactive", "draft")
_STATUS_CHECK = "status IN (%s)" % ", ".join(f"'{s}'" for s in _STATUS_VALUES)

_DIRECTION_WIDE = "direction IN ('inbound', 'outbound', 'bidirectional')"
_DIRECTION_NARROW = "direction IN ('inbound', 'outbound')"

_WELCOME_TYPE_CHECK = (
    "welcome_message_type IS NULL OR "
    "welcome_message_type IN ('user_initiated', 'ai_dynamic', 'ai_custom')"
)


# ─────────────────────────────────────────── helpers ──────────────────────────

def _has_column(conn, table: str, column: str) -> bool:
    return conn.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = :tbl AND column_name = :col)"
        ),
        {"tbl": table, "col": column},
    ).scalar()


def _has_index(conn, index_name: str) -> bool:
    return conn.execute(
        sa.text("SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = :ix)"),
        {"ix": index_name},
    ).scalar()


def _has_constraint(conn, table: str, constraint: str) -> bool:
    return conn.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.table_constraints "
            "WHERE table_schema = 'public' AND table_name = :tbl AND constraint_name = :con)"
        ),
        {"tbl": table, "con": constraint},
    ).scalar()


def _has_extension(conn, ext: str) -> bool:
    return conn.execute(
        sa.text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = :ext)"),
        {"ext": ext},
    ).scalar()


# ──────────────────────────────────────────── upgrade ─────────────────────────

def upgrade() -> None:
    conn = op.get_bind()

    # ── 0. pgcrypto extension ─────────────────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    # ── 1. agent: smart_callback column ──────────────────────────────────────
    if not _has_column(conn, "agent", "smart_callback"):
        op.add_column(
            "agent",
            sa.Column(
                "smart_callback",
                sa.Boolean(),
                nullable=False,
                server_default="false",
            ),
        )

    # ── 2. agent: ix_agent_tenant_id ─────────────────────────────────────────
    if not _has_index(conn, "ix_agent_tenant_id"):
        op.create_index("ix_agent_tenant_id", "agent", ["tenant_id"], unique=False)

    # ── 3. agent: status server_default → 'pending' for new rows ─────────────
    op.alter_column(
        "agent",
        "status",
        existing_type=sa.String(length=20),
        existing_nullable=False,
        server_default="pending",
    )

    # ── 4. agent: widened status CHECK (add if missing) ──────────────────────
    if not _has_constraint(conn, "agent", "ck_agent_status_v2"):
        op.create_check_constraint("ck_agent_status_v2", "agent", _STATUS_CHECK)

    # ── 5. agent: llm_model CHECK (nullable-safe) ─────────────────────────────
    if not _has_constraint(conn, "agent", "ck_agent_llm_model"):
        op.create_check_constraint("ck_agent_llm_model", "agent", _llm_check_sql())

    # ── 6. callflow: ix_callflow_agent_id ────────────────────────────────────
    if not _has_index(conn, "ix_callflow_agent_id"):
        op.create_index("ix_callflow_agent_id", "callflow", ["agent_id"], unique=False)

    # ── 7. callflow: widen direction CHECK ───────────────────────────────────
    if _has_constraint(conn, "callflow", "ck_callflow_direction"):
        op.drop_constraint("ck_callflow_direction", "callflow", type_="check")
    op.create_check_constraint("ck_callflow_direction", "callflow", _DIRECTION_WIDE)

    # ── 8. callflow: welcome_message_type CHECK ───────────────────────────────
    if not _has_constraint(conn, "callflow", "ck_callflow_welcome_message_type"):
        op.create_check_constraint(
            "ck_callflow_welcome_message_type", "callflow", _WELCOME_TYPE_CHECK
        )

    # ── 9. promptversion: simple flow_id index ───────────────────────────────
    if not _has_index(conn, "ix_promptversion_flow_id"):
        op.create_index("ix_promptversion_flow_id", "promptversion", ["flow_id"], unique=False)

    # ── 10. ElevenLabs key re-encryption: JWT → pgcrypto ─────────────────────
    _remigrate_elevenlabs_keys(conn)


def _remigrate_elevenlabs_keys(conn) -> None:
    """Decrypt JWT-encrypted BYO keys and re-encrypt with pgp_sym_encrypt.

    Rows that are already NULL, empty, or not legacy JWT are skipped.
    Rows whose JWT decryption fails are left untouched and counted in a warning.
    If any legacy JWT rows exist and ELEVENLABS_ENCRYPTION_KEY is unset, upgrade
    raises RuntimeError so production never ends up with a mixed ciphertext format.
    """
    try:
        from app.core.config import settings as _settings
        from app.core.db_encryption import is_legacy_jwt_ciphertext
        from app.core.security import decrypt_api_key as _jwt_decrypt
    except ImportError as exc:
        raise RuntimeError(
            "schema_v2: cannot import app settings for ElevenLabs key re-encryption. "
            "Ensure PYTHONPATH includes the project root and run alembic from the repo root."
        ) from exc

    rows = conn.execute(
        sa.text(
            "SELECT id::text, encrypted_elevenlabs_api_key "
            "FROM agent WHERE encrypted_elevenlabs_api_key IS NOT NULL"
        )
    ).fetchall()

    jwt_rows = [
        (row_id, ciphertext)
        for row_id, ciphertext in rows
        if ciphertext and is_legacy_jwt_ciphertext(ciphertext)
    ]

    enc_key = getattr(_settings, "ELEVENLABS_ENCRYPTION_KEY", "") or ""
    if jwt_rows and not enc_key:
        raise RuntimeError(
            "schema_v2: ELEVENLABS_ENCRYPTION_KEY is required before upgrade — "
            f"{len(jwt_rows)} agent row(s) still have legacy JWT-encrypted BYO ElevenLabs keys. "
            "Set the key (Secret Manager in staging/production), then re-run: alembic upgrade head"
        )

    if not jwt_rows:
        _log.info("schema_v2 key re-encryption: no legacy JWT rows — skipped")
        return

    migrated = failed = skipped = 0
    for row_id, ciphertext in rows:
        if not ciphertext or not is_legacy_jwt_ciphertext(ciphertext):
            skipped += 1
            continue
        try:
            plaintext = _jwt_decrypt(ciphertext)
            if not plaintext:
                skipped += 1
                continue
            conn.execute(
                sa.text(
                    "UPDATE agent "
                    "SET encrypted_elevenlabs_api_key = "
                    "  encode(pgp_sym_encrypt(:pt, :key), 'base64') "
                    "WHERE id = :id::uuid"
                ),
                {"pt": plaintext, "key": enc_key, "id": row_id},
            )
            migrated += 1
        except Exception as exc:
            _log.warning(
                "schema_v2: could not re-encrypt agent %s elevenlabs key: %s", row_id, exc
            )
            failed += 1

    _log.info(
        "schema_v2 key re-encryption: migrated=%d skipped=%d failed=%d",
        migrated,
        skipped,
        failed,
    )
    if failed:
        _log.warning(
            "schema_v2: %d agent row(s) still have JWT-encrypted keys; "
            "decrypt manually and re-run migration.",
            failed,
        )


# ─────────────────────────────────────────── downgrade ────────────────────────

def downgrade() -> None:
    conn = op.get_bind()

    # ElevenLabs BYO keys are NOT reversed on downgrade — see docs/db/schema-v2.md
    # "Rollback (downgrade)". Tenants must re-enter API keys after rollback.

    # ── 9. promptversion flow_id index ───────────────────────────────────────
    if _has_index(conn, "ix_promptversion_flow_id"):
        op.drop_index("ix_promptversion_flow_id", table_name="promptversion")

    # ── 8. callflow: welcome_message_type CHECK ───────────────────────────────
    if _has_constraint(conn, "callflow", "ck_callflow_welcome_message_type"):
        op.drop_constraint("ck_callflow_welcome_message_type", "callflow", type_="check")

    # ── 7. callflow: restore narrow direction CHECK ───────────────────────────
    if _has_constraint(conn, "callflow", "ck_callflow_direction"):
        op.drop_constraint("ck_callflow_direction", "callflow", type_="check")
    op.create_check_constraint("ck_callflow_direction", "callflow", _DIRECTION_NARROW)

    # ── 6. callflow: agent_id index ───────────────────────────────────────────
    if _has_index(conn, "ix_callflow_agent_id"):
        op.drop_index("ix_callflow_agent_id", table_name="callflow")

    # ── 5. agent: llm_model CHECK ────────────────────────────────────────────
    if _has_constraint(conn, "agent", "ck_agent_llm_model"):
        op.drop_constraint("ck_agent_llm_model", "agent", type_="check")

    # ── 4. agent: status CHECK ───────────────────────────────────────────────
    if _has_constraint(conn, "agent", "ck_agent_status_v2"):
        op.drop_constraint("ck_agent_status_v2", "agent", type_="check")

    # ── 3. agent: status server_default → 'active' ───────────────────────────
    op.alter_column(
        "agent",
        "status",
        existing_type=sa.String(length=20),
        existing_nullable=False,
        server_default="active",
    )

    # ── 2. agent: ix_agent_tenant_id ─────────────────────────────────────────
    if _has_index(conn, "ix_agent_tenant_id"):
        op.drop_index("ix_agent_tenant_id", table_name="agent")

    # ── 1. agent: smart_callback ─────────────────────────────────────────────
    if _has_column(conn, "agent", "smart_callback"):
        op.drop_column("agent", "smart_callback")

    # pgcrypto extension is intentionally NOT dropped — it may be used by other
    # parts of the schema (gen_random_uuid relies on pgcrypto on PG < 13).
