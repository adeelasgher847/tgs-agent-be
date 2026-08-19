"""widen ck_agent_llm_model to include gemini live native-audio models

Revision ID: 20260817_widen_llm_check_glive
Revises: 20260816_widen_llm_check_g3
Create Date: 2026-08-17

Purpose
-------
``app/core/llm_models.py`` (the Python-level allow-list used to validate
``Agent.llm_model`` at the API layer) now includes three new Google Gemini
Live speech-to-speech native-audio models: ``gemini-live-2.5-flash-native-audio``,
``gemini-live-2.5-flash-preview-native-audio-09-2025``, and
``gemini-3.1-flash-live-preview``. The DB-level ``ck_agent_llm_model`` CHECK
constraint — most recently widened by ``20260816_widen_llm_check_g3`` to a
21-value snapshot — was never widened to match, so inserting/updating an
agent with any of these new model IDs would violate the CHECK even though
the application layer accepts them.

This migration is purely additive: it drops and recreates
``ck_agent_llm_model`` with the union of the prior 21-value snapshot plus
the 3 new gemini-live IDs (24 total). No values are removed or renamed, so no
existing row can violate the widened constraint — safe to apply without a
backfill on a live multi-tenant table. The drop+recreate briefly requires a
table-level lock while Postgres validates the new CHECK against existing
rows, but on a boolean/string membership check on `agent` (not a huge
table) this is expected to be fast; the definition change itself is
otherwise the same additive-CHECK-widening pattern used by
``20260816_widen_llm_check_g3``, ``20260815_widen_llm_check``, and, before
them, ``20260602_schema_v2_completion`` for ``ck_agent_status_v2`` /
``ck_callflow_direction``.

Note: these three model IDs select an entirely different call pipeline
(speech-to-speech, no external STT/TTS) — this migration only widens the
CHECK so the value is storable; it does not change any pipeline behavior.

Snapshot strategy
------------------
Per the same rationale as ``20260816_widen_llm_check_g3``, the allow-list
used to build the CHECK SQL is a self-contained snapshot tuple in *this*
file, not an import from ``app.core.llm_models`` — so future edits to that
module (renames, removals) do not retroactively change what this historical
migration replays during ``alembic upgrade``.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260817_widen_llm_check_glive"
down_revision: Union[str, Sequence[str], None] = "20260816_widen_llm_check_g3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Union of the 20260816_widen_llm_check_g3 21-value snapshot plus the 3 new
# Gemini Live native-audio model IDs added to
# app.core.llm_models.ALLOWED_LLM_MODELS. Self-contained — do not import
# app.core.llm_models (future renames there must not retroactively change
# this migration's replay behavior).
_ALLOWED_LLM_MODELS_AT_REVISION: tuple[str, ...] = (
    # --- prior 21-value snapshot (20260816_widen_llm_check_g3), unchanged ---
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
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    # --- new: Google Gemini Live native-audio family ---
    "gemini-live-2.5-flash-native-audio",
    "gemini-live-2.5-flash-preview-native-audio-09-2025",
    "gemini-3.1-flash-live-preview",
)

# The exact 21-value snapshot from 20260816_widen_llm_check_g3, needed to
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
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
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
