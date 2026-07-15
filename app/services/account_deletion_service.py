"""
GDPR right-to-erasure: irreversible workspace account deletion.

Per ticket technical notes, the PII wipe runs as a single transaction of raw
SQL UPDATE/DELETE statements (no ORM) for clarity and auditability — the SQL
below is the literal, reviewable list of every column this operation touches.

Vector embeddings: RAG embeddings live exclusively in Postgres via pgvector
(KbChunk.embedding, see app/models/knowledge_base_chunk.py) — there is no
Pinecone or other external vector store in this codebase (rag_service.py
replaced the old Pinecone backend). The `DELETE FROM kbchunk` below removes
the embedding column along with the rest of the row, so no separate
vector-store erasure step is needed for GDPR compliance.
"""
from __future__ import annotations

import uuid

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.logger import logger

_ANON_EMAIL = "[DELETED@DELETED.COM]"
_ANON_NAME = "[DELETED]"
_ANON_PHONE = "[REDACTED]"


def delete_workspace_account(db: Session, workspace_id: uuid.UUID) -> None:
    """
    Wipe PII for a workspace and soft-delete it. Irreversible.

    Users who belong ONLY to this workspace have their identity columns
    anonymized and are soft-deleted. Users who also belong to other
    (non-deleted) workspaces keep their identity intact — it is shared data
    outside the scope of this erasure request — but lose membership in this
    one. Without this distinction, anonymizing a shared user's email would
    lock them out of unrelated tenants they still belong to.
    """
    params = {"workspace_id": str(workspace_id)}

    try:
        # Lets the no_update_audit trigger anonymize actor_* columns on
        # auditlog for this transaction only (see migration
        # 20260618_audit_update_bypass). auditlog rows are never deleted.
        db.execute(text("SET LOCAL app.bypass_audit_update = 'true'"))

        db.execute(
            text(
                "UPDATE auditlog SET user_id = NULL, actor_api_key_prefix = '[DELETED]' "
                "WHERE tenant_id = :workspace_id"
            ),
            params,
        )

        db.execute(
            text(
                """
                CREATE TEMP TABLE _gdpr_sole_tenant_users ON COMMIT DROP AS
                SELECT tu.user_id
                FROM (
                    SELECT user_id FROM user_tenant_association WHERE tenant_id = :workspace_id
                ) tu
                WHERE NOT EXISTS (
                    SELECT 1 FROM user_tenant_association other
                    WHERE other.user_id = tu.user_id AND other.tenant_id != :workspace_id
                )
                """
            ),
            params,
        )

        db.execute(
            text(
                """
                UPDATE "user"
                SET email = :anon_email,
                    first_name = :anon_name,
                    last_name = :anon_name,
                    phone = NULL,
                    deleted_at = now()
                WHERE id IN (SELECT user_id FROM _gdpr_sole_tenant_users)
                """
            ),
            {**params, "anon_email": _ANON_EMAIL, "anon_name": _ANON_NAME},
        )

        # Multi-tenant users keep their identity; only their membership in
        # *this* workspace is removed.
        db.execute(
            text(
                """
                DELETE FROM user_tenant_association
                WHERE tenant_id = :workspace_id
                  AND user_id NOT IN (SELECT user_id FROM _gdpr_sole_tenant_users)
                """
            ),
            params,
        )

        db.execute(
            text("UPDATE phonenumber SET phone_number = :anon_phone WHERE tenant_id = :workspace_id"),
            {**params, "anon_phone": _ANON_PHONE},
        )

        db.execute(
            text(
                "UPDATE callsession SET from_number = :anon_phone, to_number = :anon_phone "
                "WHERE tenant_id = :workspace_id"
            ),
            {**params, "anon_phone": _ANON_PHONE},
        )

        db.execute(
            text(
                "DELETE FROM kbchunk WHERE kb_id IN "
                "(SELECT id FROM knowledgebase WHERE workspace_id = :workspace_id)"
            ),
            params,
        )

        db.execute(
            text("UPDATE tenant SET deleted_at = now() WHERE id = :workspace_id"),
            params,
        )

        db.commit()
    except Exception:
        db.rollback()
        raise

    try:
        from app.services.s3_recording_service import delete_workspace_recordings

        delete_workspace_recordings(workspace_id)
    except Exception as exc:
        # PII in Postgres is already wiped and committed (the core legal
        # obligation); an S3 hiccup here shouldn't undo that or fail the
        # otherwise-successful deletion. Logged loudly for ops follow-up.
        logger.error(
            "account_deletion: S3 recording cleanup failed for workspace %s: %s",
            workspace_id, exc, exc_info=True,
        )
