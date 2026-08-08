"""
Provider-capability layer for the humanization engine
(app.voice.humanization_engine).

Answers "what can this TTS provider's actual, currently-used code path do"
without scattering `if provider == "elevenlabs"` checks through decision
logic. Capabilities are read from what app.utils.tts_adapter's
synthesize/stream_synthesize/async_stream_synthesize methods and their
underlying services (elevenlabs_service, google_tts_service,
rime_tts_service) genuinely do today — not from provider documentation.

Phase 3B scope: capability lookup + a pure settings-overlay helper that
folds a HumanizationDecision into a provider's existing voice-settings dict,
using only parameters the adapter already consumes (no new parameters
invented). Nothing in this module is called from any call-flow file yet —
wiring ElevenLabsAdapter/TtsPipeline/the handlers to actually use it is a
later integration phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from app.voice.humanization_engine import HumanizationDecision

_ELEVENLABS_STABILITY_MIN = 0.0
_ELEVENLABS_STABILITY_MAX = 1.0


@dataclass(frozen=True)
class TTSProviderCapabilities:
    """
    What a provider's live (streaming-path) code genuinely supports today.

    `supports_ssml` reflects the streaming path specifically — the path
    nearly every utterance takes (VOICE_TTS_STREAM_MIN_WORDS=2 means
    streaming triggers at just 2 words). All three providers strip SSML
    there today (see tts_stream_mixin.py's _stream_tts_chunk and
    google_tts_service.stream_text_to_speech's defensive tag-stripping),
    even though Google's rarely-used non-streaming batch text_to_speech
    path does honor `<speak>`-prefixed SSML.
    """

    provider_slug: str
    supports_streaming: bool
    supports_ssml: bool
    supports_speaking_rate: bool
    supports_pitch: bool
    supports_stability_control: bool
    supports_native_expressive_tags: bool
    supports_pause_control: bool


_UNKNOWN_PROVIDER_CAPABILITIES = TTSProviderCapabilities(
    provider_slug="unknown",
    supports_streaming=False,
    supports_ssml=False,
    supports_speaking_rate=False,
    supports_pitch=False,
    supports_stability_control=False,
    supports_native_expressive_tags=False,
    supports_pause_control=False,
)

_CAPABILITIES: Dict[str, TTSProviderCapabilities] = {
    # ElevenLabsAdapter.async_stream_synthesize (app/utils/tts_adapter.py:219)
    "elevenlabs": TTSProviderCapabilities(
        provider_slug="elevenlabs",
        supports_streaming=True,
        supports_ssml=False,  # stripped every call in tts_stream_mixin.py's _stream_tts_chunk
        supports_speaking_rate=True,  # voice_settings["speed"] (elevenlabs_service._default_voice_settings)
        supports_pitch=False,  # no pitch parameter anywhere in the ElevenLabs request payload
        supports_stability_control=True,  # voice_settings["stability"]/"similarity_boost"/"style"
        supports_native_expressive_tags=True,  # bracket tags, app/utils/eleven_tts_text.py
        supports_pause_control=False,  # no native break/pause param used; SSML <break> stripped
    ),
    # google_tts_service.stream_text_to_speech (app/services/google_tts_service.py:330)
    "google": TTSProviderCapabilities(
        provider_slug="google",
        supports_streaming=True,
        supports_ssml=False,  # streaming path always strips tags (google_tts_service.py:381-388)
        supports_speaking_rate=True,  # GoogleTTSAdapter._resolve_speaking_rate
        supports_pitch=False,  # GoogleTTSAdapter.stream_synthesize never forwards pitch (batch-only param)
        supports_stability_control=False,  # no stability concept for Google
        supports_native_expressive_tags=False,  # bracket tags are stripped for non-ElevenLabs providers
        supports_pause_control=False,  # SSML <break> stripped on the streaming path
    ),
    # RimeTTSAdapter.async_stream_synthesize (app/utils/tts_adapter.py:434)
    "rime": TTSProviderCapabilities(
        provider_slug="rime",
        supports_streaming=True,  # sync stream_synthesize explicitly raises NotImplementedError
        supports_ssml=False,  # Rime's API has no SSML concept at all
        supports_speaking_rate=True,  # speed -> speedAlpha via _user_speed_to_speed_alpha
        supports_pitch=False,
        supports_stability_control=False,
        supports_native_expressive_tags=False,
        supports_pause_control=False,  # reduceLatency=True actively trims inter-sentence silence
    ),
}


def get_capabilities(provider_slug: str | None) -> TTSProviderCapabilities:
    """
    Look up capabilities for a provider slug. Unknown/missing providers get
    a fully-False capability set rather than an exception, so callers can
    always safely gate on `caps.supports_x` without special-casing
    "unknown" — normal provider behavior is never blocked by this lookup.
    """
    slug = (provider_slug or "").strip().lower()
    return _CAPABILITIES.get(slug, _UNKNOWN_PROVIDER_CAPABILITIES)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_voice_settings_overlay(
    provider_slug: str | None,
    decision: HumanizationDecision | None,
) -> Dict[str, Any]:
    """
    Translate a HumanizationDecision into a provider voice-settings overlay
    dict, using only keys the provider's existing adapter/service code
    already consumes (no new parameters invented). A future integration
    phase would merge this dict OVER a provider's existing settings_json;
    this function has no side effects and is not called from any
    call-flow file in this phase.

    Capability-gated, not provider-name-gated: this dispatches on
    `caps.supports_stability_control`, not `provider_slug == "elevenlabs"`,
    so a future provider gaining the same capability needs no change here.

    Returns {} whenever there is nothing safe to apply, so a caller that
    merges this overlay in always preserves the provider's existing default
    behavior unchanged (e.g. ElevenLabsService._default_voice_settings()'s
    stability=0.45 stays untouched when no valid hint is available).
    """
    overlay: Dict[str, Any] = {}
    if decision is None:
        return overlay

    caps = get_capabilities(provider_slug)

    if caps.supports_stability_control:
        hint = decision.tts_stability_hint
        if isinstance(hint, (int, float)) and not isinstance(hint, bool):
            overlay["stability"] = _clamp(
                float(hint), _ELEVENLABS_STABILITY_MIN, _ELEVENLABS_STABILITY_MAX
            )
        # Missing/invalid hint: no "stability" key added, so the caller's
        # existing default (or explicit settings_json override) wins.

    return overlay
