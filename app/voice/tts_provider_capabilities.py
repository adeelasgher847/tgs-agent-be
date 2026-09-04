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
from app.voice.humanization_intent import DeliveryEmotion, VocalBehavior

_ELEVENLABS_STABILITY_MIN = 0.0
_ELEVENLABS_STABILITY_MAX = 1.0

# V-08: conservative ElevenLabs stability deltas per DeliveryEmotion,
# extending the existing single-axis mood->stability dial
# (turn_signals._tts_stability_for_mood) to the richer, LLM-driven enum.
# Kept in the SAME 0.30-0.85 "consistency/expressiveness" band that dial
# already documents — no aggressive swings, UPBEAT nudges down slightly
# (more dynamic) rather than toward an unstable extreme.
_ELEVEN_STABILITY_BY_EMOTION: Dict[DeliveryEmotion, float] = {
    DeliveryEmotion.NEUTRAL: 0.50,
    DeliveryEmotion.WARM: 0.55,
    DeliveryEmotion.CALM: 0.60,
    DeliveryEmotion.APOLOGETIC: 0.58,
    DeliveryEmotion.UPBEAT: 0.45,
}

# V-08: small, conservative speaking-rate nudges for providers with a
# `speaking_rate`/`speed` knob but no stability concept (Google, Rime, Hume,
# xAI) — consolidates the previously-duplicated inline emotion->rate maps
# in app.voice.tts_stream_mixin (two call sites) and
# app.voice.livekit_browser_call_handler into this one function.
_SPEAKING_RATE_BY_EMOTION: Dict[DeliveryEmotion, float] = {
    DeliveryEmotion.NEUTRAL: 1.0,
    DeliveryEmotion.WARM: 0.99,
    DeliveryEmotion.CALM: 0.97,
    DeliveryEmotion.APOLOGETIC: 0.96,
    DeliveryEmotion.UPBEAT: 1.03,
}

# Pre-V-08 fallback: the EXACT values previously duplicated inline in
# app.voice.tts_stream_mixin (two call sites) keyed off
# app.utils.tts_preprocessing.detect_emotion's plain-string response_emotion
# classification (happy/sad/uncertain/confident) — preserved verbatim here
# so consolidating those call sites into this function does not change
# existing (VOICE_ENABLE_LLM_HUMANIZATION=off) behavior.
_SPEAKING_RATE_BY_RESPONSE_EMOTION: Dict[str, float] = {
    "happy": 1.03,
    "sad": 0.97,
    "uncertain": 0.98,
    "confident": 1.01,
}

# V-08: ElevenLabs-only native bracket-tag inners for each VocalBehavior,
# reusing app.utils.eleven_tts_text's existing _ELEVEN_V3_TAG_INNERS
# whitelist verbatim (chuckles/sighs/hesitates are already whitelisted
# there) rather than inventing a parallel tag vocabulary.
_VOCAL_BEHAVIOR_TAG_INNER: Dict[VocalBehavior, str] = {
    VocalBehavior.SOFT_CHUCKLE: "chuckles",
    VocalBehavior.BRIEF_SIGH: "sighs",
    VocalBehavior.HESITATION: "hesitates",
}


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
    # Phase 6-3: whether a persistent, incremental WebSocket `stream-input`
    # synthesis session (app.services.elevenlabs_ws_session) is available for
    # this provider, as an alternative to the default one-HTTP-request-per-
    # chunk path. Only genuinely true for ElevenLabs today — Google/Rime have
    # no equivalent persistent-session streaming-input protocol implemented.
    # Consulted by app.voice.tts_pipeline.TtsPipeline._try_elevenlabs_ws_route
    # as the actual capability gate (alongside the
    # VOICE_TTS_ELEVENLABS_STREAMING_SESSION_ENABLED flag) — never a
    # provider-name string check duplicated at the call site.
    supports_streaming_session: bool = False


_UNKNOWN_PROVIDER_CAPABILITIES = TTSProviderCapabilities(
    provider_slug="unknown",
    supports_streaming=False,
    supports_ssml=False,
    supports_speaking_rate=False,
    supports_pitch=False,
    supports_stability_control=False,
    supports_native_expressive_tags=False,
    supports_pause_control=False,
    supports_streaming_session=False,
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
        # Phase 6-2 spike (scripts/spikes/elevenlabs_websocket_spike.py) empirically
        # validated the `stream-input` WebSocket protocol (persistent session,
        # incremental SendText, auto_mode, flush-at-turn-end, clean cancellation).
        supports_streaming_session=True,
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
        supports_streaming_session=False,  # no persistent streaming-input protocol for Google here
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
        supports_streaming_session=False,  # no persistent streaming-input protocol for Rime here
    ),
    # HumeTTSAdapter.async_stream_synthesize (app/utils/tts_adapter.py)
    "hume": TTSProviderCapabilities(
        provider_slug="hume",
        supports_streaming=True,  # sync stream_synthesize explicitly raises NotImplementedError
        supports_ssml=False,  # Hume's WebSocket protocol has no SSML concept — plain text only
        supports_speaking_rate=True,  # settings_json["speed"] -> Hume's native "speed" param (0.5-2.0)
        supports_pitch=False,  # no pitch parameter in the Hume WS payload
        supports_stability_control=False,  # Hume's prosody knob is "description" (free-text acting
        # instructions), a structurally different mechanism from ElevenLabs' numeric
        # stability/similarity_boost — build_voice_settings_overlay() is explicitly scoped
        # to the numeric path and is not wired to Hume's description field
        supports_native_expressive_tags=False,  # bracket tags are stripped for non-ElevenLabs providers
        supports_pause_control=False,  # no native break/pause param used
        supports_streaming_session=False,  # one-shot WS request per chunk, not a persistent session
    ),
    # XaiTTSAdapter.async_stream_synthesize (app/utils/tts_adapter.py)
    "xai": TTSProviderCapabilities(
        provider_slug="xai",
        supports_streaming=True,  # sync stream_synthesize explicitly raises NotImplementedError
        supports_ssml=False,  # xAI's WebSocket protocol has no SSML concept — plain text.delta only
        supports_speaking_rate=True,  # settings_json["speed"] -> xAI's native "speed" query param (0.7-1.5)
        supports_pitch=False,  # no pitch parameter documented for xAI's TTS WebSocket
        supports_stability_control=False,  # no stability/similarity_boost-style numeric knob documented
        supports_native_expressive_tags=False,  # bracket tags are stripped for non-ElevenLabs providers
        supports_pause_control=False,  # no native break/pause param used
        supports_streaming_session=False,  # one-shot WS request per chunk, not a persistent session
        # (text.clear/audio.clear exist for a future persistent-session mode
        # but are unused by this integration — see xai_tts_service.py)
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
    delivery = getattr(decision, "delivery", None)

    if caps.supports_stability_control:
        # V-08: an LLM-requested delivery emotion takes precedence over the
        # caller-mood-derived tts_stability_hint for THIS segment (still
        # capability-gated, still clamped to the same safe band) — but only
        # when there IS a delivery decision with a non-neutral-default
        # emotion; NEUTRAL/absent falls through to the pre-existing hint so
        # behavior is unchanged for every call that never enables V-08.
        if (
            delivery is not None
            and delivery.emotion != DeliveryEmotion.NEUTRAL
            and delivery.emotion in _ELEVEN_STABILITY_BY_EMOTION
        ):
            overlay["stability"] = _clamp(
                _ELEVEN_STABILITY_BY_EMOTION[delivery.emotion],
                _ELEVENLABS_STABILITY_MIN,
                _ELEVENLABS_STABILITY_MAX,
            )
        else:
            hint = decision.tts_stability_hint
            if isinstance(hint, (int, float)) and not isinstance(hint, bool):
                overlay["stability"] = _clamp(
                    float(hint), _ELEVENLABS_STABILITY_MIN, _ELEVENLABS_STABILITY_MAX
                )
            # Missing/invalid hint: no "stability" key added, so the
            # caller's existing default (or explicit settings_json
            # override) wins.
    elif caps.supports_speaking_rate:
        # V-08: providers with a rate knob but no stability concept
        # (Google/Rime/Hume/xAI). Consolidates what used to be two
        # independently-duplicated inline emotion->rate maps
        # (app.voice.tts_stream_mixin ~lines 959-970/1384-1390 and
        # app.voice.livekit_browser_call_handler's hardcoded 1.0) into this
        # one function, replacing all three call sites.
        #
        # An explicit, non-neutral LLM delivery emotion takes precedence
        # (same conservative deltas as the ElevenLabs stability branch
        # above); otherwise this falls back to the SAME
        # detect_emotion()-derived response_emotion heuristic and the SAME
        # literal rate values (happy=1.03/sad=0.97/uncertain=0.98/
        # confident=1.01) the old duplicated inline code used — so a
        # deployment with VOICE_ENABLE_LLM_HUMANIZATION off (the default)
        # sees byte-identical rate nudges to before this consolidation.
        if (
            delivery is not None
            and delivery.emotion != DeliveryEmotion.NEUTRAL
            and delivery.emotion in _SPEAKING_RATE_BY_EMOTION
        ):
            rate = _SPEAKING_RATE_BY_EMOTION[delivery.emotion]
            if rate != 1.0:
                overlay["speaking_rate"] = rate
        else:
            rate = _SPEAKING_RATE_BY_RESPONSE_EMOTION.get(decision.response_emotion)
            if rate is not None and rate != 1.0:
                overlay["speaking_rate"] = rate

    return overlay


def apply_vocal_behavior_tag(
    text: str,
    provider_slug: str | None,
    decision: HumanizationDecision | None,
) -> str:
    """
    Realize `decision.delivery.vocal_behavior` (V-08) as a native
    ElevenLabs bracket tag prefix (e.g. "[chuckles] ..."), reusing
    `app.utils.eleven_tts_text`'s existing `_ELEVEN_V3_TAG_INNERS`
    whitelist — never a new tag vocabulary.

    Capability-gated via `caps.supports_native_expressive_tags`: for every
    OTHER provider (Google, Rime, Hume, xAI — none of which have a native
    audio-tag concept) this is a no-op, and DELIBERATELY never simulates
    the effect with literal spoken text like "*sighs*". Returns `text`
    unchanged whenever there is nothing safe to apply (no decision, no
    delivery, NONE behavior, or an unsupported provider).
    """
    if not text or decision is None:
        return text
    delivery = getattr(decision, "delivery", None)
    if delivery is None or delivery.vocal_behavior == VocalBehavior.NONE:
        return text

    caps = get_capabilities(provider_slug)
    if not caps.supports_native_expressive_tags:
        return text

    tag_inner = _VOCAL_BEHAVIOR_TAG_INNER.get(delivery.vocal_behavior)
    if not tag_inner:
        return text

    return f"[{tag_inner}] {text}"


def apply_emphasis_word(
    text: str,
    provider_slug: str | None,
    decision: HumanizationDecision | None,
) -> str:
    """
    V-08 `emphasis_word` realization: intentionally a no-op for EVERY
    provider today, including ElevenLabs.

    Investigated and rejected: ElevenLabs' bracket-tag mechanism
    (`app.utils.eleven_tts_text`) only carries whole-phrase mood/action
    tags (e.g. "[excited]", "[sighs]") — there is no documented, reliable
    native mechanism for stressing a single word within an otherwise
    unmarked sentence via that same channel, and SSML `<emphasis>` is
    stripped from every provider's live streaming path today (see
    `TTSProviderCapabilities.supports_ssml`, False everywhere). Per this
    feature's explicit instructions, a fake effect (e.g. wrapping the word
    in asterisks/caps that would be read aloud literally) must never be
    invented as a substitute — so this stays a documented no-op until a
    provider adds a real per-word emphasis primitive to its streaming path.
    """
    return text
