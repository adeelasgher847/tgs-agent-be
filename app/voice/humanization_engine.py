"""
Centralized, provider-agnostic humanization decision layer.

Given the assistant's text for a turn (plus light turn context), produces a
HumanizationDecision describing how it should be expressed more naturally —
final tone-adapted text, pacing hints, acknowledgement eligibility,
tone/emotion signal, and a still-gated filler hint. This module makes no
LLM/network calls, does not sleep, and never touches audio, SSML, or any TTS
provider — it only classifies and applies deterministic, cosmetic text
shaping. Wiring provider-specific controls onto its output is separate work
(see app.voice.tts_provider_capabilities).

Deliberately reuses rather than reimplements:
- app.voice.turn_signals.build_turn_context for mood, response-brevity, and
  the tts_stability_hint (consumed by tts_provider_capabilities).
- app.voice.tone_adapter.tone_adapter for the actual text shaping — the
  SAME pure, deterministic function previously called directly from
  app/routers/bidirectional_stream.py (Twilio only). It is provider- and
  transport-agnostic (depends only on text/TurnContext/use_ssml, no
  Twilio-specific state), so centralizing the *call site* here — instead of
  duplicating its logic — gives both Twilio and the browser/LiveKit path the
  same tone adaptation from one place. `HumanizationDecision.text` is now
  this function's output: the single source of truth for "final TTS text"
  that TtsPipeline consumes, rather than each caller applying its own
  independent text shaping before/after this decision.
- app.voice.conversation_orchestrator.should_send_quick_ack /
  QuickAckConfig for acknowledgement eligibility rules. Only eligibility is
  reused here — phrase selection and probabilistic sampling stay owned by
  conversation_orchestrator.send_quick_acknowledgement.
- app.utils.tts_preprocessing.detect_emotion for the response text's own
  emotional coloring (distinct axis from the caller's inferred mood).

Fillers stay disabled by design: app.utils.tts_preprocessing.insert_fillers
was already turned off upstream because of audio clicking artifacts. This
module only defines a typed place for that decision to live later; it does
not re-enable filler injection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from app.core.config import settings
from app.utils.tts_preprocessing import detect_emotion
from app.voice.humanization_intent import PauseCategory, SegmentIntent, VocalBehavior
from app.voice.tone_adapter import tone_adapter
from app.voice.tts_flush import SENTENCE_END_RE, SOFT_BOUNDARY_RE
from app.voice.turn_signals import UserMood

_SHORT_UTTERANCE_MAX_WORDS = 6

# ── V-08: LLM-driven delivery guardrails (deterministic, never the LLM's call) ──
#
# Cooldown: a requested vocal_behavior is suppressed if the previous ONE was
# used within this many segments (a "segment" == one analyze_response() call,
# i.e. one flushed TTS chunk) — so a chuckle/sigh/hesitation can't repeat
# back-to-back or every other sentence.
_VOCAL_BEHAVIOR_COOLDOWN_SEGMENTS = 4
# Hard ceiling on total non-NONE vocal_behavior realizations for the whole
# call — a "handful", not unlimited, regardless of how often the LLM asks.
_VOCAL_BEHAVIOR_CALL_CEILING = 6


@dataclass
class VocalBehaviorGuardrailState:
    """
    Per-call, in-memory-only counters enforcing the vocal_behavior
    guardrails above. Lives on the transport handler instance (already the
    per-active-call, in-memory object per this repo's convention — see
    CLAUDE.md's "never hold conversation/call state in memory across
    requests"; this is explicitly NOT cross-request state, it is ephemeral
    delivery flavor for the CURRENT live call only, discarded on disconnect
    exactly like the handler's other in-memory TTS/STT buffers). Never
    persisted — a mid-call process restart simply resets the cooldown/
    ceiling, which is an acceptable (and intentional, per the task spec)
    trade-off for state this ephemeral.
    """

    last_vocal_behavior: VocalBehavior = VocalBehavior.NONE
    segments_since_last_vocal_behavior: int = _VOCAL_BEHAVIOR_COOLDOWN_SEGMENTS
    total_vocal_behavior_used: int = 0

    def record(self, behavior: VocalBehavior) -> None:
        """Advance counters after a segment's (possibly guardrail-downgraded)
        vocal_behavior has been decided."""
        if behavior != VocalBehavior.NONE:
            self.last_vocal_behavior = behavior
            self.segments_since_last_vocal_behavior = 0
            self.total_vocal_behavior_used += 1
        else:
            self.segments_since_last_vocal_behavior += 1


def _apply_delivery_guardrails(
    intent: SegmentIntent,
    state: Optional[VocalBehaviorGuardrailState],
) -> SegmentIntent:
    """
    Deterministically enforce the vocal_behavior cooldown/ceiling guardrails
    on an already-validated `SegmentIntent`. The LLM's request is advisory
    only — this function has the final say, always, regardless of the
    LLM's own `confidence`. Never raises: any unexpected error degrades to
    passing `intent` through unchanged (worst case: one extra vocal
    behavior slips through on a genuinely unexpected error, never a crash
    or a dropped/blocked response).

    `emphasis_word` substring validity and "at most one vocal_behavior per
    segment" are already guaranteed by `SegmentIntent`'s own construction
    (`humanization_intent.build_segment_intent`) — not re-validated here.
    """
    try:
        behavior = intent.vocal_behavior
        if behavior != VocalBehavior.NONE and state is not None:
            if (
                state.segments_since_last_vocal_behavior
                < _VOCAL_BEHAVIOR_COOLDOWN_SEGMENTS
            ):
                behavior = VocalBehavior.NONE
            elif state.total_vocal_behavior_used >= _VOCAL_BEHAVIOR_CALL_CEILING:
                behavior = VocalBehavior.NONE

        if state is not None:
            state.record(behavior)

        if behavior == intent.vocal_behavior:
            return intent
        return SegmentIntent(
            text=intent.text,
            emotion=intent.emotion,
            vocal_behavior=behavior,
            pause_after=intent.pause_after,
            emphasis_word=intent.emphasis_word,
            confidence=intent.confidence,
        )
    except Exception:
        return intent


class SentenceEndingType(str, Enum):
    """How a chunk's text currently ends — a partial/mid-stream chunk with
    no terminal punctuation yet is NONE, not an error."""

    STATEMENT = "statement"
    QUESTION = "question"
    EXCLAMATION = "exclamation"
    NONE = "none"


@dataclass(frozen=True)
class PacingHint:
    """
    Provider- and audio-agnostic hints about a chunk's conversational shape:
    sentence count/boundaries, short-utterance flag, how it currently ends,
    and whether an internal pause would be natural. Metadata only — no
    delay, silence, SSML <break>, or audio change is produced by this module
    or by anything that currently consumes a HumanizationDecision. A future
    audio-layer phase decides whether/how to act on these hints.
    """

    sentence_count: int = 0
    has_multiple_sentences: bool = False
    is_short_utterance: bool = True
    ending_type: SentenceEndingType = SentenceEndingType.NONE
    has_internal_pause_opportunity: bool = False


@dataclass(frozen=True)
class AcknowledgementHint:
    """
    Whether a backchannel/acknowledgement is eligible for this turn.

    Eligibility only, reusing conversation_orchestrator.should_send_quick_ack.
    Phrase choice and probability sampling are intentionally NOT duplicated
    here — they remain the responsibility of the existing send path.
    """

    eligible: bool = False


@dataclass(frozen=True)
class FillerHint:
    """
    Filler-word (umm/uh/hmm) decision. Always `allowed=False` in this phase:
    tts_preprocessing.insert_fillers() was already disabled upstream due to
    audio clicking artifacts, and this phase must not re-enable it. Exists
    so a future, audio-tested filler feature has a typed decision to plug
    into instead of inventing a parallel mechanism.
    """

    allowed: bool = False


@dataclass(frozen=True)
class HumanizationDecision:
    """
    Provider-neutral humanization decision for one assistant text chunk.

    `text` is the single source of truth for "final TTS text": the input
    passed through tone_adapter() (cosmetic tone shaping only — leading
    "chipper" openers stripped, a handful of exclamatory stock phrases
    softened when the caller's mood is sad/frustrated/angry/urgent; never
    touches names, numbers, URLs, or any factual content). When humanization
    is disabled, input is empty, or anything fails, `text` falls back to the
    original, completely unmodified input.
    """

    text: str
    mood: UserMood
    response_emotion: str
    pacing: PacingHint
    acknowledgement: AcknowledgementHint
    filler: FillerHint
    tts_stability_hint: float | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # V-08: guardrailed LLM delivery intent for this segment, or None when
    # VOICE_ENABLE_LLM_HUMANIZATION is off, no llm_intent was supplied, or
    # parsing/guardrail evaluation failed. Provider realization
    # (app.voice.tts_provider_capabilities) reads this field; nothing about
    # `text`/`mood`/`pacing`/etc above changes based on its presence.
    delivery: SegmentIntent | None = None


def _neutral_decision(text: str) -> HumanizationDecision:
    """
    Safe "no humanization" fallback. Used when the engine is disabled, given
    empty/falsy input, or hits any unexpected error — the original text
    always remains usable and TTS is never blocked.
    """
    return HumanizationDecision(
        text=text or "",
        mood=UserMood.NEUTRAL,
        response_emotion="neutral",
        pacing=PacingHint(),
        acknowledgement=AcknowledgementHint(),
        filler=FillerHint(),
    )


def _pacing_hint(text: str) -> PacingHint:
    """
    Reuses app.voice.tts_flush's punctuation-aware boundary regexes (the
    same ones that decide LLM-stream flush points) rather than a naive
    `[.!?]` scan, so a bare "." inside a decimal ("42.50"), a URL
    ("example.com"), or a dotted phone number ("555.123.4567") is correctly
    NOT counted as a sentence boundary — those patterns require the
    punctuation to be followed by whitespace or end-of-string.
    """
    sentence_count = sum(1 for _ in SENTENCE_END_RE.finditer(text))
    words = text.split()

    stripped = text.rstrip()
    if not stripped or stripped[-1] not in ".!?":
        # No terminal punctuation yet — a partial/mid-stream chunk, not an error.
        ending_type = SentenceEndingType.NONE
    elif stripped[-1] == "?":
        ending_type = SentenceEndingType.QUESTION
    elif stripped[-1] == "!":
        ending_type = SentenceEndingType.EXCLAMATION
    else:
        ending_type = SentenceEndingType.STATEMENT

    has_multiple = sentence_count > 1
    has_pause_opportunity = has_multiple or bool(SOFT_BOUNDARY_RE.search(text))

    return PacingHint(
        sentence_count=sentence_count,
        has_multiple_sentences=has_multiple,
        is_short_utterance=len(words) <= _SHORT_UTTERANCE_MAX_WORDS,
        ending_type=ending_type,
        has_internal_pause_opportunity=has_pause_opportunity,
    )


def _acknowledgement_hint(user_text: str) -> AcknowledgementHint:
    # Deferred import: conversation_orchestrator does not import this module,
    # so this is not strictly circular, but keeping it deferred matches the
    # existing codebase convention (see bidirectional_stream.py's deferred
    # VertexGeminiService import) and keeps this module cheap to import.
    from app.voice.conversation_orchestrator import (
        VOICE_TUNABLES,
        should_send_quick_ack,
    )

    eligible = should_send_quick_ack(user_text, VOICE_TUNABLES.quick_ack)
    return AcknowledgementHint(eligible=eligible)


# V-08: provider-neutral silence-frame counts (20ms each) for each semantic
# PauseCategory the LLM can request. Same mechanism as the pre-existing
# pacing-based pause below (literal 0xFF silence frames via the transport's
# existing paced-send path) — never a new audio primitive.
_PAUSE_CATEGORY_FRAMES: Dict[PauseCategory, int] = {
    PauseCategory.NONE: 0,
    PauseCategory.BREATH: 3,
    PauseCategory.THINKING: 6,
    PauseCategory.EMPHASIS: 2,
}


def pause_frames_for_chunk(
    pacing: PacingHint | None,
    is_final: bool,
    *,
    pause_after: PauseCategory | None = None,
) -> int:
    """
    Phase 4C-2: how many extra 0xFF silence frames (20ms each) should follow
    this chunk's real audio, or 0 for none. Pure metadata decision — this
    function never touches audio, never sleeps, and is not itself on any
    async path. Callers (TtsStreamMixin._stream_tts_chunk,
    LiveKitBrowserCallHandler._stream_tts_chunk) execute the returned count
    using their own existing paced-send mechanism; this only decides "how
    many", never "how".

    `pause_after` (V-08, optional, keyword-only): when the LLM's delivery
    intent requested a specific `PauseCategory` for this segment
    (BREATH/THINKING/EMPHASIS) AND `VOICE_ENABLE_LLM_HUMANIZATION` is on,
    that request takes precedence over the regex-derived pacing heuristic
    below (still never on a final chunk). Passing None (the default — no
    LLM intent, or LLM humanization disabled) preserves the EXACT
    pre-existing pacing-only behavior, byte-for-byte.

    Eligibility for the pacing-only path (all must hold, else 0):
    - `pacing` is present (not None — e.g. humanization disabled/failed)
    - VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES > 0 (default 0 = always inert)
    - `is_final` is False (the existing end-of-turn silence drain already
      handles true utterance end — never stack both)
    - `pacing.ending_type` is a real sentence/question/exclamation end, not
      NONE (never pause after a mid-sentence/time-based fallback flush)
    - `pacing.is_short_utterance` is False (never pause after a short
      acknowledgement-style fragment)

    Never raises: any unexpected error (malformed `pacing`, bad config
    value) degrades to 0, so a pacing failure can never affect playback.
    """
    try:
        if is_final:
            return 0

        if (
            pause_after is not None
            and pause_after != PauseCategory.NONE
            and bool(getattr(settings, "VOICE_ENABLE_LLM_HUMANIZATION", False))
        ):
            return _PAUSE_CATEGORY_FRAMES.get(pause_after, 0)

        if pacing is None:
            return 0
        configured = int(
            getattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 0) or 0
        )
        if configured <= 0:
            return 0
        if pacing.ending_type == SentenceEndingType.NONE:
            return 0
        if pacing.is_short_utterance:
            return 0
        return configured
    except Exception:
        return 0


def analyze_response(
    text: str,
    *,
    user_text: str = "",
    stt_confidence: float = 0.0,
    booking_context_active: bool = False,
    is_final: bool = True,
    use_ssml: bool = False,
    llm_intent: SegmentIntent | None = None,
    guardrail_state: VocalBehaviorGuardrailState | None = None,
) -> HumanizationDecision:
    """
    Produce a HumanizationDecision for one assistant response/chunk.

    Cheap, synchronous, in-memory only — safe to call on the hot LLM-to-TTS
    streaming path (no LLM/network calls, no sleeps, no queues). Never
    raises: on a disabled feature flag, empty input, or any unexpected
    error (including a tone_adapter failure), this degrades to a neutral
    "no humanization" decision — `text` falls back to the original,
    completely unmodified input — so callers can always fall back to text
    that's safe to speak and TTS is never blocked.

    `use_ssml` must match the same flag the caller will pass to
    TtsPipeline.queue_tts()/the TTS provider for this chunk: tone_adapter's
    substitutions are a no-op whenever use_ssml is True (its calling
    convention, unchanged here) — passing a mismatched value would silently
    diverge from what the caller intended.

    `llm_intent` (V-08, optional): an already-validated `SegmentIntent`
    parsed by `app.voice.humanization_intent` from this segment's inline
    `[DELIVERY ...]` tag. Only consulted when
    `VOICE_ENABLE_LLM_HUMANIZATION` is True; otherwise (flag off, or this
    argument is None — e.g. parsing failed upstream) the returned
    `HumanizationDecision.delivery` is always None and every other field is
    computed EXACTLY as before this parameter existed — this is the
    regression contract callers rely on. When present and the flag is on,
    its emotion/vocal_behavior/pause_after take precedence over what the
    regex-based path would otherwise imply for THIS segment's delivery —
    caller-mood detection (`turn_signals.detect_mood`, feeding
    `turn_ctx.mood`/`tts_stability_hint` above) is unaffected either way.
    Deterministic guardrails (cooldown/ceiling) are always enforced here,
    never left to the LLM's own request.

    `guardrail_state` (V-08, optional): per-call `VocalBehaviorGuardrailState`
    used to enforce the vocal_behavior cooldown/ceiling across the whole
    call. Passing None (e.g. a caller that hasn't wired call-scoped state)
    skips cooldown/ceiling enforcement for this single call — callers on
    the live streaming path always pass a real, per-call instance.
    """
    if not bool(getattr(settings, "VOICE_ENABLE_HUMANIZATION_ENGINE", True)):
        return _neutral_decision(text)

    try:
        safe_text = (text or "").strip()
        if not safe_text:
            return _neutral_decision(text or "")

        from app.voice.turn_signals import build_turn_context

        turn_ctx = build_turn_context(
            user_text,
            stt_confidence,
            booking_context_active=booking_context_active,
            is_final=is_final,
        )

        final_text = tone_adapter(safe_text, turn_ctx, use_ssml) or safe_text

        # Isolated fallback: a pacing-analysis failure degrades only pacing
        # to neutral, rather than discarding the whole decision (mood, tone
        # adaptation, ack eligibility, stability hint) over what is, at
        # worst, a regex edge case on untrusted text.
        #
        # Computed from final_text (post tone_adapter), not safe_text: pacing
        # must describe the text that will actually be synthesized/spoken —
        # tone_adapter can strip a leading chipper opener or substitute
        # exclamatory stock phrases, which can change word count/sentence
        # boundaries enough to flip is_short_utterance/ending_type relative
        # to the pre-adaptation text.
        try:
            pacing = _pacing_hint(final_text)
        except Exception:
            pacing = PacingHint()

        # V-08: LLM-driven delivery — reachable ONLY when the flag is on AND
        # a caller-supplied llm_intent is present, so a disabled deployment
        # (or a caller that never passes llm_intent) never executes this
        # branch and `delivery` stays None, matching pre-V-08 output exactly.
        delivery: SegmentIntent | None = None
        if isinstance(llm_intent, SegmentIntent) and bool(
            getattr(settings, "VOICE_ENABLE_LLM_HUMANIZATION", False)
        ):
            try:
                delivery = _apply_delivery_guardrails(llm_intent, guardrail_state)
            except Exception:
                delivery = None

        return HumanizationDecision(
            text=final_text,
            mood=turn_ctx.mood,
            response_emotion=detect_emotion(safe_text),
            pacing=pacing,
            acknowledgement=_acknowledgement_hint(user_text),
            filler=FillerHint(),
            tts_stability_hint=turn_ctx.tts_stability_hint,
            delivery=delivery,
            metadata={
                "conversation_phase": turn_ctx.conversation_phase,
                "respond_briefly": turn_ctx.respond_briefly,
            },
        )
    except Exception:
        return _neutral_decision(text or "")
