"""widen ck_agent_llm_model to include gemini-3 family

Revision ID: 20260816_widen_llm_check_g3
Revises: 20260815_widen_llm_check
Create Date: 2026-08-16

Purpose
-------
``app/core/llm_models.py`` (the Python-level allow-list used to validate
``Agent.llm_model`` at the API layer) now includes two new Google Gemini
models: ``gemini-3-flash-preview`` (Gemini 3 Flash Preview, Vertex AI
preview) and ``gemini-3.1-flash-lite`` (Gemini 3.1 Flash-Lite, GA as of
May 2026). The DB-level ``ck_agent_llm_model`` CHECK constraint — most
recently widened by ``20260815_widen_llm_check`` to a 19-value snapshot —
was never widened to match, so inserting/updating an agent with either of
these new model IDs would violate the CHECK even though the application
layer accepts them.

This migration is purely additive: it drops and recreates
``ck_agent_llm_model`` with the union of the prior 19-value snapshot plus
the 2 new gemini-3 IDs (21 total). No values are removed or renamed, so no
existing row can violate the widened constraint — safe to apply without a
backfill on a live multi-tenant table. The drop+recreate briefly requires a
table-level lock while Postgres validates the new CHECK against existing
rows, but on a boolean/string membership check on `agent` (not a huge
table) this is expected to be fast; the definition change itself is
otherwise the same additive-CHECK-widening pattern used by
``20260815_widen_llm_check`` and, before it, ``20260602_schema_v2_completion``
for ``ck_agent_status_v2`` / ``ck_callflow_direction``.

Snapshot strategy
------------------
Per the same rationale as ``20260815_widen_llm_check``, the allow-list used
to build the CHECK SQL is a self-contained snapshot tuple in *this* file,
not an import from ``app.core.llm_models`` — so future edits to that module
(renames, removals) do not retroactively change what this historical
migration replays during ``alembic upgrade``.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260816_widen_llm_check_g3"
down_revision: Union[str, Sequence[str], None] = "20260815_widen_llm_check"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Union of the 20260815_widen_llm_check 19-value snapshot plus the 2 new
# Gemini 3 family IDs added to app.core.llm_models.ALLOWED_LLM_MODELS.
# Self-contained — do not import app.core.llm_models (future renames there
# must not retroactively change this migration's replay behavior).
_ALLOWED_LLM_MODELS_AT_REVISION: tuple[str, ...] = (
    # --- prior 19-value snapshot (20260815_widen_llm_check), unchanged ---
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4-turbo",
    "gemini-2.5-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "claude-3-5-sonnet",
    "claude-3-haiku",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.4",
    # --- new: Google Gemini 3 family ---
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
)

# The exact 19-value snapshot from 20260815_widen_llm_check, needed to
# restore the narrower constraint on downgrade.
_PRIOR_ALLOWED_LLM_MODELS: tuple[str, ...] = (
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4-turbo",
    "gemini-2.5-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "claude-3-5-sonnet",
    "claude-3-haiku",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.4",
)


def _llm_check_sql(models: tuple[str, ...]) -> str:
    """Build ck_agent_llm_model CHECK SQL from a given model snapshot."""
    return "llm_model IS NULL OR llm_model IN (%s)" % ", ".join(
        f"'{m}'" for m in models
    )


def _has_constraint(conn, table: str, constraint: str) -> bool:
    return conn.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.table_constraints "
            "WHERE table_schema = 'public' AND table_name = :tbl AND constraint_name = :con)"
        ),
        {"tbl": table, "con": constraint},
    ).scalar()


def upgrade() -> None:
    conn = op.get_bind()

    if _has_constraint(conn, "agent", "ck_agent_llm_model"):
        op.drop_constraint("ck_agent_llm_model", "agent", type_="check")
    op.create_check_constraint(
        "ck_agent_llm_model", "agent", _llm_check_sql(_ALLOWED_LLM_MODELS_AT_REVISION)
    )


def downgrade() -> None:
    conn = op.get_bind()

    if _has_constraint(conn, "agent", "ck_agent_llm_model"):
        op.drop_constraint("ck_agent_llm_model", "agent", type_="check")
    op.create_check_constraint(
        "ck_agent_llm_model", "agent", _llm_check_sql(_PRIOR_ALLOWED_LLM_MODELS)
    )
