"""
Allowed LLM models for agent configuration.

Single source of truth for the ``llm_model`` field on agents. Read by the
``/api/v1/agent`` endpoints to validate requests and to build the
``allowedValues`` array returned with ``invalid_llm_model`` errors.

Add/remove a model by editing :data:`ALLOWED_LLM_MODELS` only — do not
hardcode model identifiers anywhere else.  The Alembic migration
``20260602_schema_v2_completion`` snapshots the allow-list at revision time;
add a new migration when the DB CHECK must widen to match this tuple.
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
    # OpenAI — GPT-5 reasoning family (new; NOT set as any existing agent's
    # default — see app/services/openai_service.py for the temperature /
    # max_completion_tokens handling this family requires).
    "gpt-5",
    "gpt-5-mini",
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.4",
    # Google Gemini — ticket required + existing
    "gemini-2.5-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    # Google Gemini — Gemini 3 family (new)
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    # Google Gemini — Live/native-audio speech-to-speech models (new). These
    # are NOT text models — selecting one routes the whole call through a
    # completely different pipeline (caller audio in, Gemini's native audio
    # out, no external STT/TTS). See GEMINI_LIVE_MODELS below for the
    # per-model auth/location config and is_gemini_live_native_audio_model()
    # for the routing predicate. `infer_llm_provider` still returns "gemini"
    # for these (they start with "gemini") — no change needed there.
    "gemini-live-2.5-flash-native-audio",
    "gemini-live-2.5-flash-preview-native-audio-09-2025",
    "gemini-3.1-flash-live-preview",
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


#: Per-model auth/location config for the Gemini Live native-audio family.
#: Confirmed live against the real API (2026-08-17) — each model has a
#: *different* auth/location requirement, so this MUST stay a per-model
#: table rather than collapsing to one global auth mode:
#:
#: - ``gemini-live-2.5-flash-native-audio`` / \
#:   ``gemini-live-2.5-flash-preview-native-audio-09-2025``: Vertex AI / ADC,
#:   regional ``us-central1`` only (confirmed NOT available on ``global`` —
#:   the opposite restriction from the Gemini 3 text family in
#:   ``vertex_gemini_service.py``'s ``_GLOBAL_ONLY_MODELS``).
#: - ``gemini-3.1-flash-live-preview``: Gemini Developer API / API key
#:   (``settings.GEMINI_API_KEY``) — confirmed NOT available on Vertex AI for
#:   this project in any region (404 across 4 locations x 6 model-ID
#:   spellings). This is a deliberate, scoped exception to this codebase's
#:   otherwise Vertex/ADC-only convention for Gemini models — approved by the
#:   user. The other two models must never read ``GEMINI_API_KEY``.
GEMINI_LIVE_MODELS: Final[dict[str, dict[str, str | None]]] = {
    "gemini-live-2.5-flash-native-audio": {"auth": "vertex", "location": "us-central1"},
    "gemini-live-2.5-flash-preview-native-audio-09-2025": {
        "auth": "vertex",
        "location": "us-central1",
    },
    "gemini-3.1-flash-live-preview": {"auth": "api_key", "location": None},
}


def is_gemini_live_native_audio_model(model_name: str) -> bool:
    """
    Return True if ``model_name`` is one of the Gemini Live speech-to-speech
    native-audio models (:data:`GEMINI_LIVE_MODELS`).

    Exact-match check — these models require a completely different call
    pipeline (caller audio in, Gemini's own native audio out, no external
    STT/TTS), not just another text-generation ``gemini-*`` model routed
    through the usual streaming-text pipeline. Do not loosen this to a
    prefix/substring match: other ``gemini-*``/``gemini-live-*`` strings
    that are not in the allow-list must not be silently treated as
    native-audio.
    """
    return (model_name or "") in GEMINI_LIVE_MODELS


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
