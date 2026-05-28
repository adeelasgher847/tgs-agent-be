"""
Resolve agent ticket fields for voice/LLM runtime.

Ticket CRUD stores ``llm_model`` and ``ttsModel`` (``tts_provider_slug``, etc.).
Call paths use these helpers first, then fall back to legacy ``model_id`` /
``tts_provider_id`` relations when ticket fields are absent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from app.core.llm_models import infer_llm_provider
from app.core.logger import logger
from app.core.security import decrypt_api_key
from app.models.agent import Agent


_TICKET_TTS_TO_ADAPTER: dict[str, str] = {
    "11labs": "elevenlabs",
    "11labs_byo": "elevenlabs",
    # "rime" now has its own adapter — no longer falls back to google.
    "rime": "rime",
}


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
    # Rime has a built-in default voice — allow ticket path even without explicit voice_id.
    has_voice = bool(agent.tts_voice_external_id) or (
        (agent.tts_provider_slug or "").lower() == "rime"
    )
    return bool(agent.tts_provider_slug and agent.tts_language and has_voice)


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

    temperature = 0.15
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
        if agent.model and agent.model.api_key:
            try:
                api_key = decrypt_api_key(agent.model.api_key)
            except Exception as exc:
                logger.error("Failed to decrypt legacy model API key: %s", exc)
        return ResolvedLlmRuntime(
            model_name=agent.llm_model,
            provider_slug=provider_slug,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            used_ticket_llm=True,
        )

    if agent.model:
        provider_slug = "gemini"
        if agent.provider:
            pname = (agent.provider.name or "").lower()
            if "openai" in pname:
                provider_slug = "openai"
            elif "groq" in pname:
                provider_slug = "groq"
            elif "gemini" in pname or "google" in pname:
                provider_slug = "gemini"
        api_key = None
        if agent.model.api_key:
            try:
                api_key = decrypt_api_key(agent.model.api_key)
            except Exception as exc:
                logger.error("Failed to decrypt agent model API key: %s", exc)
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

    if agent.language:
        language = agent.language
    settings = dict(agent.tts_settings_json or {})

    if _ticket_tts_triad(agent):
        slug = (agent.tts_provider_slug or "").lower()
        adapter_slug = _TICKET_TTS_TO_ADAPTER.get(slug, slug)
        voice_id = agent.tts_voice_external_id
        if agent.tts_language:
            language = agent.tts_language
        if slug == "11labs_byo" and agent.encrypted_elevenlabs_api_key:
            try:
                settings["elevenlabs_api_key"] = decrypt_api_key(
                    agent.encrypted_elevenlabs_api_key
                )
            except Exception as exc:
                logger.error("Failed to decrypt BYO ElevenLabs key for agent %s: %s", agent.id, exc)
        # Normalize speed + volume for all providers (default 1.0 if absent).
        settings.setdefault("speed", float(settings.get("speed", 1.0)))
        settings.setdefault("volume", float(settings.get("volume", 1.0)))
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
        settings.setdefault("speed", float(settings.get("speed", 1.0)))
        settings.setdefault("volume", float(settings.get("volume", 1.0)))
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
