"""
Resolve agent ticket fields for voice/LLM runtime.

Ticket CRUD stores ``llm_model`` and ``ttsModel`` (``tts_provider_slug``, etc.).
Call paths use these helpers first, then fall back to legacy ``model_id`` /
``tts_provider_id`` relations when ticket fields are absent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union
from uuid import UUID

from typing import TYPE_CHECKING

from app.core.config import settings as app_settings
from app.core.llm_models import infer_llm_provider
from app.core.logger import logger
from app.core.security import decrypt_api_key
from app.models.agent import Agent

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


_TICKET_TTS_TO_ADAPTER: dict[str, str] = {
    "11labs": "elevenlabs",
    "11labs_byo": "elevenlabs",
    # "rime" now has its own adapter — no longer falls back to google.
    "rime": "rime",
}

def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _merge_nested_tts_settings(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Accept both flat and nested shapes for tts_settings_json.

    Flat (preferred): {"speed": 0.8, "volume": 1.0, ...}
    Nested (UI/legacy): {"settings": {"speed": 0.8, "volume": 1.0}, ...}

    Nested keys are merged into the top level WITHOUT overwriting an explicit
    top-level value (top level wins on conflict).
    """
    merged = dict(raw or {})
    nested = merged.get("settings")
    if isinstance(nested, dict):
        for k, v in nested.items():
            merged.setdefault(k, v)
    return merged


@dataclass(frozen=True)
class ResolvedLlmRuntime:
    model_name: str
    provider_slug: str
    api_key: Optional[str]
    temperature: float
    max_tokens: int
    used_ticket_llm: bool


@dataclass(frozen=True)
class ResolvedTtsRuntime:
    adapter_slug: str
    voice_external_id: Optional[str]
    language: str
    settings_json: dict[str, Any]
    used_ticket_tts: bool


def _decrypt_stored_api_key(
    encrypted: str,
    *,
    agent_id: Union[UUID, str, None],
    credential_label: str,
) -> str:
    """Decrypt a stored API key; fail fast with a clear operator-facing error."""
    try:
        return decrypt_api_key(encrypted)
    except Exception as exc:
        logger.error(
            "Failed to decrypt %s for agent %s: %s",
            credential_label,
            agent_id,
            exc,
            exc_info=True,
        )
        raise RuntimeError(
            f"Agent {agent_id}: stored {credential_label} is corrupted or unreadable. "
            "Re-save the API key in agent settings."
        ) from exc


def _ticket_tts_triad(agent: Agent) -> bool:
    # Rime has a built-in default voice — allow ticket path even without explicit voice_id.
    has_voice = bool(agent.tts_voice_external_id) or (
        (agent.tts_provider_slug or "").lower() == "rime"
    )
    return bool(agent.tts_provider_slug and agent.tts_language and has_voice)


def _is_gemini_provider(provider_slug: str) -> bool:
    return (provider_slug or "").lower() == "gemini"


def resolve_llm_runtime(agent: Optional[Agent]) -> ResolvedLlmRuntime:
    """Pick LLM model + provider for conversation / scheduling paths."""
    _default_temperature = float(
        getattr(app_settings, "VOICE_LLM_DEFAULT_TEMPERATURE", 0.3)
    )
    default = ResolvedLlmRuntime(
        model_name=app_settings.DEFAULT_LLM_MODEL,
        provider_slug=app_settings.DEFAULT_LLM_PROVIDER,
        api_key=None,
        temperature=_default_temperature,
        max_tokens=100,
        used_ticket_llm=False,
    )
    if not agent:
        return default

    temperature = _default_temperature
    max_tokens = 100
    if agent.agent_temperature is not None:
        temperature = agent.agent_temperature / 100.0
    elif agent.model and agent.model.temperature is not None:
        temperature = agent.model.temperature / 100.0
    if agent.agent_max_tokens:
        max_tokens = agent.agent_max_tokens
    elif agent.model and agent.model.max_tokens:
        max_tokens = agent.model.max_tokens

    if agent.llm_model:
        provider_slug = infer_llm_provider(agent.llm_model)
        api_key: Optional[str] = None
        # Gemini/Google models authenticate via ADC (GOOGLE_APPLICATION_CREDENTIALS).
        # Never use model.api_key for the voice Vertex path — ignore it even if set.
        if not _is_gemini_provider(provider_slug) and agent.model and agent.model.api_key:
            api_key = _decrypt_stored_api_key(
                agent.model.api_key,
                agent_id=agent.id,
                credential_label="LLM model API key",
            )
        return ResolvedLlmRuntime(
            model_name=agent.llm_model,
            provider_slug=provider_slug,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            used_ticket_llm=True,
        )

    if agent.model:
        provider_slug = app_settings.DEFAULT_LLM_PROVIDER
        if agent.provider:
            pname = (agent.provider.name or "").lower()
            if "openai" in pname:
                provider_slug = "openai"
            elif "groq" in pname:
                provider_slug = "groq"
            elif "gemini" in pname or "google" in pname:
                provider_slug = "gemini"
        api_key = None
        # Gemini/Google: skip model.api_key — auth via ADC
        if not _is_gemini_provider(provider_slug) and agent.model.api_key:
            api_key = _decrypt_stored_api_key(
                agent.model.api_key,
                agent_id=agent.id,
                credential_label="LLM model API key",
            )
        return ResolvedLlmRuntime(
            model_name=agent.model.model_name,
            provider_slug=provider_slug,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            used_ticket_llm=False,
        )

    return ResolvedLlmRuntime(
        model_name=default.model_name,
        provider_slug=default.provider_slug,
        api_key=None,
        temperature=temperature,
        max_tokens=max_tokens,
        used_ticket_llm=False,
    )


def llm_service_for_provider(provider_slug: str) -> Any:
    """Return the shared LLM service instance for a provider slug.

    Gemini/Google → VertexGeminiService (ADC auth, correct system_instruction).
    OpenAI / Groq → unchanged.
    gemini_service is kept for RAG embeddings only (embed_text path).
    """
    from app.services.groq_service import groq_service
    from app.services.openai_service import openai_service
    from app.services.vertex_gemini_service import vertex_gemini_service

    slug = (provider_slug or "").lower()
    if slug == "openai":
        return openai_service
    if slug == "groq":
        return groq_service
    # Default: gemini / google → Vertex AI path
    return vertex_gemini_service


def resolve_tts_runtime(
    agent: Optional[Agent],
    db: "Session | None" = None,
) -> ResolvedTtsRuntime:
    """Map ticket ``ttsModel`` (or legacy relations) to adapter + voice id.

    ``db`` is used only when the agent uses a BYO ElevenLabs key stored with
    pgcrypto encryption.  Pass the caller's existing session when available;
    if omitted and needed a short-lived session is opened automatically.
    """
    # Read bounds at call time (not import) — defaults live in Settings (config.py).
    tts_speed_min = app_settings.TTS_SPEED_MIN
    tts_speed_max = app_settings.TTS_SPEED_MAX
    tts_volume_min = app_settings.TTS_VOLUME_MIN
    tts_volume_max = app_settings.TTS_VOLUME_MAX

    language = "en"
    settings: dict[str, Any] = {}

    if not agent:
        return ResolvedTtsRuntime(
            adapter_slug="google",
            voice_external_id=None,
            language=language,
            settings_json=settings,
            used_ticket_tts=False,
        )

    if agent.language:
        language = agent.language
    settings = _merge_nested_tts_settings(agent.tts_settings_json or {})

    # Normalise + clamp user-facing speed/volume so downstream adapters can
    # rely on safe ranges. Uniform semantics across all providers:
    #   speed: 1.0 = normal, 0.8 = slower, 1.2 = faster
    #   volume: 1.0 = normal, 0.0 = silence, 2.0 = max louder
    speed = _clamp(
        _coerce_float(settings.get("speed", 1.0), 1.0), tts_speed_min, tts_speed_max
    )
    volume = _clamp(
        _coerce_float(settings.get("volume", 1.0), 1.0), tts_volume_min, tts_volume_max
    )
    settings["speed"] = speed
    settings["volume"] = volume

    if _ticket_tts_triad(agent):
        slug = (agent.tts_provider_slug or "").lower()
        adapter_slug = _TICKET_TTS_TO_ADAPTER.get(slug, slug)
        if slug == "rime" and adapter_slug == "google":
            logger.warning(
                "Agent %s: Rime TTS not yet available — falling back to Google TTS",
                agent.id,
            )
        voice_id = agent.tts_voice_external_id
        if agent.tts_language:
            language = agent.tts_language
        if slug == "11labs_byo" and agent.encrypted_elevenlabs_api_key:
            try:
                from app.core.db_encryption import decrypt_stored_elevenlabs_key
                settings["elevenlabs_api_key"] = decrypt_stored_elevenlabs_key(
                    agent.encrypted_elevenlabs_api_key,
                    db=db,
                )
            except Exception as exc:
                logger.error(
                    "Failed to decrypt BYO ElevenLabs API key for agent %s: %s",
                    agent.id,
                    exc,
                    exc_info=True,
                )
                raise RuntimeError(
                    f"Agent {agent.id}: stored BYO ElevenLabs API key is corrupted or "
                    "unreadable. Re-save the API key in agent settings."
                ) from exc
        # For Rime: ensure default voice when none configured.
        if adapter_slug == "rime" and not voice_id:
            voice_id = "mistv2_Wildflower"
        settings.setdefault("language_code", language)
        return ResolvedTtsRuntime(
            adapter_slug=adapter_slug,
            voice_external_id=voice_id,
            language=language,
            settings_json=settings,
            used_ticket_tts=True,
        )

    legacy_provider = getattr(agent, "tts_provider", None)
    if legacy_provider and getattr(legacy_provider, "slug", None):
        adapter_slug = (legacy_provider.slug or "google").lower()
        tts_voice = getattr(agent, "tts_voice", None)
        voice_id = getattr(tts_voice, "external_voice_id", None) if tts_voice else None
        return ResolvedTtsRuntime(
            adapter_slug=adapter_slug,
            voice_external_id=voice_id,
            language=language,
            settings_json=settings,
            used_ticket_tts=False,
        )

    return ResolvedTtsRuntime(
        adapter_slug="google",
        voice_external_id=None,
        language=language,
        settings_json=settings,
        used_ticket_tts=False,
    )


def resolve_tts_adapter_slug(
    agent: Optional[Agent],
    db: "Session | None" = None,
) -> Optional[str]:
    """Convenience for call sites that only need the adapter slug string."""
    return resolve_tts_runtime(agent, db=db).adapter_slug
