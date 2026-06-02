"""
Allowed LLM models for agent configuration.

Single source of truth for the ``llm_model`` field on agents. Read by the
``/api/v1/agent`` endpoints to validate requests and to build the
``allowedValues`` array returned with ``invalid_llm_model`` errors.

Add/remove a model by editing :data:`ALLOWED_LLM_MODELS` only — do not
hardcode model identifiers anywhere else.  The Alembic migration
``20260602_schema_v2_completion`` builds ``ck_agent_llm_model`` from this tuple
via ``_llm_check_sql()`` in that revision file.
"""
from __future__ import annotations

from typing import Final

ALLOWED_LLM_MODELS: Final[tuple[str, ...]] = (
    # OpenAI — ticket required + existing
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4-turbo",
    # Google Gemini — ticket required + existing
    "gemini-2.5-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    # Anthropic — existing
    "claude-3-5-sonnet",
    "claude-3-haiku",
    # Groq — existing
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
)


def is_allowed_llm_model(model: str) -> bool:
    """Return True if ``model`` is in the allow-list (case-sensitive)."""
    return model in ALLOWED_LLM_MODELS


def allowed_llm_models() -> list[str]:
    """Return a fresh list copy — safe to embed in JSON responses."""
    return list(ALLOWED_LLM_MODELS)


def infer_llm_provider(model_name: str) -> str:
    """
    Infer runtime provider slug from an allow-listed ``llm_model`` string.

    Used when agents are configured via ticket fields without a legacy
    ``provider_id`` / ``model`` row.
    """
    name = (model_name or "").strip().lower()
    if name.startswith("gpt") or name.startswith("o1") or name.startswith("o3"):
        return "openai"
    if name.startswith("gemini"):
        return "gemini"
    if name.startswith("claude"):
        return "gemini"  # no Anthropic service yet — same model id may fail at API
    if name.startswith("llama"):
        return "groq"
    return "gemini"
