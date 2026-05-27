"""
Resolve agent ticket fields for voice/LLM runtime.

Ticket CRUD stores ``llm_model`` and ``ttsModel`` (``tts_provider_slug``, etc.).
Call paths use these helpers first, then fall back to legacy relations when
ticket fields are absent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from app.core.llm_models import infer_llm_provider
from app.core.logger import logger
from app.core.security import decrypt_api_key
from app.models.agent import Agent
from app.schemas.agent import normalize_tts_provider_slug


_TICKET_TTS_TO_ADAPTER: dict[str, str] = {
    "elevenlabs": "elevenlabs",
    "elevenlabs_byo": "elevenlabs",
    "11labs": "elevenlabs",
    "11labs_byo": "elevenlabs",
    "rime": "google",  # no Rime adapter yet — safe telephony fallback
}


def _provider_slug_from_agent(agent: Agent) -> str:
    if agent.provider and agent.provider.name:
        pname = agent.provider.name.lower()
        if "openai" in pname:
            return "openai"
        if "groq" in pname:
            return "groq"
        if "gemini" in pname or "google" in pname:
            return "gemini"
        if "anthropic" in pname or "claude" in pname:
            return "gemini"
    if agent.model and getattr(agent.model, "provider", None) and agent.model.provider.name:
        pname = agent.model.provider.name.lower()
        if "openai" in pname:
            return "openai"
        if "groq" in pname:
            return "groq"
        if "gemini" in pname or "google" in pname:
            return "gemini"
    # Ticket llm_model field takes precedence for provider inference
    return infer_llm_provider(agent.llm_model or "")


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


def _ticket_tts_triad(agent: Agent) -> bool:
    return bool(
        getattr(agent, "tts_provider_slug", None)
        and getattr(agent, "tts_voice_external_id", None)
        and getattr(agent, "tts_language", None)
    )


def resolve_llm_runtime(agent: Optional[Agent]) -> ResolvedLlmRuntime:
    """Pick LLM model + provider for conversation / scheduling paths."""
    default = ResolvedLlmRuntime(
        model_name="gemini-1.5-flash",
        provider_slug="gemini",
        api_key=None,
        temperature=0.15,
        max_tokens=100,
        used_ticket_llm=False,
    )
    if not agent:
        return default

    # Default temperature: Gemini 2.5 Flash voice path uses 0.3 for natural conversational tone;
    # other providers keep 0.15 (shorter, more deterministic voice replies).
    _default_temp = 0.3 if (agent.llm_model or "").strip() == "gemini-2.5-flash" else 0.15
    temperature = _default_temp
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
        provider_slug = _provider_slug_from_agent(agent)
        api_key: Optional[str] = None
        # Google AI Studio (GeminiService) uses API keys; decrypt model.api_key when present.
        if agent.model and agent.model.api_key:
            try:
                api_key = decrypt_api_key(agent.model.api_key)
            except Exception as exc:
                logger.error("Failed to decrypt model API key: %s", exc)
        return ResolvedLlmRuntime(
            model_name=agent.llm_model,
            provider_slug=provider_slug,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            used_ticket_llm=True,
        )

    if agent.model:
        api_key = None
        if agent.model.api_key:
            try:
                api_key = decrypt_api_key(agent.model.api_key)
            except Exception as exc:
                logger.error("Failed to decrypt agent model API key: %s", exc)
        return ResolvedLlmRuntime(
            model_name=agent.model.model_name,
            provider_slug=_provider_slug_from_agent(agent),
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
    """Return the shared LLM service instance for a provider slug."""
    from app.services.gemini_service import gemini_service
    from app.services.groq_service import groq_service
    from app.services.openai_service import openai_service

    slug = (provider_slug or "").lower()
    if slug == "openai":
        return openai_service
    if slug == "groq":
        return groq_service
    return gemini_service


def resolve_tts_runtime(agent: Optional[Agent]) -> ResolvedTtsRuntime:
    """Map ticket ``ttsModel`` (or legacy relations) to adapter + voice id."""
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

    agent_language = getattr(agent, "language", None)
    if agent_language:
        language = agent_language
    settings = dict(getattr(agent, "tts_settings_json", None) or {})

    if _ticket_tts_triad(agent):
        slug = normalize_tts_provider_slug(agent.tts_provider_slug or "")
        adapter_slug = _TICKET_TTS_TO_ADAPTER.get(slug, slug)
        if slug == "rime":
            logger.debug(
                "Agent %s uses ticket TTS provider 'rime'; falling back to google TTS adapter",
                agent.id,
            )
        voice_id = agent.tts_voice_external_id
        if agent.tts_language:
            language = agent.tts_language
        if slug in ("elevenlabs_byo", "11labs_byo") and agent.encrypted_elevenlabs_api_key:
            try:
                settings["elevenlabs_api_key"] = decrypt_api_key(
                    agent.encrypted_elevenlabs_api_key
                )
            except Exception as exc:
                logger.error("Failed to decrypt BYO ElevenLabs key for agent %s: %s", agent.id, exc)
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


def resolve_tts_adapter_slug(agent: Optional[Agent]) -> Optional[str]:
    """Convenience for call sites that only need the adapter slug string."""
    return resolve_tts_runtime(agent).adapter_slug
