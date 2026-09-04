"""
Pure, provider-agnostic parsing/validation for LLM-driven TTS delivery
intent ("Option C" humanization architecture, V-08).

This module has NO TTS/provider imports and makes no network/audio calls —
it only defines the wire format the LLM emits, and validates/normalizes
whatever comes back into a `SegmentIntent`. Realizing a `SegmentIntent`
into actual provider settings/audio (ElevenLabs bracket tags, speaking-rate
nudges, silence frames) is deliberately kept OUT of this module — see
`app.voice.tts_provider_capabilities` and `app.voice.humanization_engine`.

Wire format — inline bracket tag, not JSON-Lines
-------------------------------------------------
Both call transports (`app/routers/bidirectional_stream.py`'s inline
LLM-streaming loop and `app/voice/conversation_orchestrator.py`'s
`generate_and_stream_response`) already buffer raw LLM tokens character by
character and flush a *prefix* of that buffer to TTS as soon as
`app.voice.tts_flush.find_sentence_flush_index`/`find_time_flush_index`
finds a safe boundary — and they already recognize/strip OTHER inline
bracket control tokens emitted by the same LLM call today
(`[END_CALL]`, `[TRANSFER_CALL]`, `[OUTCOME:...]`, `[CHECK_SLOTS:...]`,
`[BOOK_APPOINTMENT:...]`). A JSON-Lines format (the audit's original
suggestion) would require each segment to arrive as one complete,
well-formed JSON object before it could be parsed — which fights the
existing token-level flush buffering, which flushes on PARTIAL text well
before a hypothetical closing `}` would arrive, and would require a
second, bespoke incremental-JSON accumulator duplicated in both transports.

Instead, delivery intent rides as ONE more inline bracket tag,
`[DELIVERY ...]`, emitted immediately BEFORE the segment of spoken text it
describes — e.g.:

    [DELIVERY emotion=warm behavior=none pause=breath] Sure, I can help
    with that. [DELIVERY emotion=neutral] Let me check your account.

This composes for free with the existing control-token stripping and
sentence/time flush logic: the tag is recognized and removed from the
buffer via the same kind of regex already used for `[OUTCOME:...]` etc.,
and the segment's plain text is exactly whatever the existing flush logic
was already going to flush — no parallel buffering mechanism, no second
LLM call, and the LLM only ever sees/emits the semantic enums below (never
milliseconds or provider-specific tag names).

Because the tag can arrive split across multiple streamed tokens, callers
must only treat a tag as present once a *complete* `[DELIVERY ...]` has
arrived (closing `]` seen) — see `extract_pending_delivery_tag`. An
incomplete tag is left untouched in the buffer to keep accumulating.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple

from app.core.config import settings


class DeliveryEmotion(str, Enum):
    NEUTRAL = "neutral"
    WARM = "warm"
    UPBEAT = "upbeat"
    CALM = "calm"
    APOLOGETIC = "apologetic"


class VocalBehavior(str, Enum):
    NONE = "none"
    SOFT_CHUCKLE = "soft_chuckle"
    BRIEF_SIGH = "brief_sigh"
    HESITATION = "hesitation"


class PauseCategory(str, Enum):
    NONE = "none"
    BREATH = "breath"
    THINKING = "thinking"
    EMPHASIS = "emphasis"


@dataclass(frozen=True)
class SegmentIntent:
    """
    Validated, provider-agnostic delivery intent for one spoken segment.

    Every field always has a safe default — a `SegmentIntent` is always
    usable as-is, even when it represents "nothing special, just say the
    text" (the common case, and the ONLY case whenever
    `VOICE_ENABLE_LLM_HUMANIZATION` is off).
    """

    text: str
    emotion: DeliveryEmotion = DeliveryEmotion.NEUTRAL
    vocal_behavior: VocalBehavior = VocalBehavior.NONE
    pause_after: PauseCategory = PauseCategory.NONE
    emphasis_word: Optional[str] = None
    confidence: float = 1.0


def _safe_enum(enum_cls: type, raw: Any, default: Any) -> Any:
    """Map any raw value to a member of `enum_cls`, or `default` — never raises."""
    if raw is None:
        return default
    try:
        key = str(raw).strip().lower()
    except Exception:
        return default
    if not key:
        return default
    for member in enum_cls:
        if member.value == key:
            return member
    return default


def _safe_confidence(raw: Any) -> float:
    try:
        value = float(raw) if raw is not None else 1.0
    except (TypeError, ValueError):
        return 1.0
    if value != value:  # NaN
        return 1.0
    return max(0.0, min(1.0, value))


def build_segment_intent(
    text: str,
    *,
    emotion: Any = None,
    vocal_behavior: Any = None,
    pause_after: Any = None,
    emphasis_word: Any = None,
    confidence: Any = None,
) -> SegmentIntent:
    """
    The one true validation entry point: turn raw (possibly malformed,
    possibly LLM-hallucinated) attribute values into a safe `SegmentIntent`.

    Never raises. Guarantees:
    - Unknown/garbage enum values silently become their NEUTRAL/NONE default.
    - Missing/empty `text` collapses the WHOLE segment to plain-text-only
      neutral defaults (still carrying whatever text WAS given, even if
      falsy/empty) — a malformed delivery tag never blocks or drops the
      actual response text, it only ever loses its own flavor.
    - `emphasis_word` is discarded (set to None) unless it is a
      case-insensitive substring of `text` — never fails the segment.
    """
    safe_text = text if isinstance(text, str) else ""
    if not safe_text.strip():
        # Invalid/absent segment text: fall back to fully neutral defaults,
        # but never invent or drop text that wasn't there — passthrough of
        # whatever (possibly empty) text was available.
        return SegmentIntent(text=safe_text)

    emo = _safe_enum(DeliveryEmotion, emotion, DeliveryEmotion.NEUTRAL)
    behavior = _safe_enum(VocalBehavior, vocal_behavior, VocalBehavior.NONE)
    pause = _safe_enum(PauseCategory, pause_after, PauseCategory.NONE)

    emphasis: Optional[str] = None
    if emphasis_word:
        try:
            candidate = str(emphasis_word).strip()
        except Exception:
            candidate = ""
        if candidate and candidate.lower() in safe_text.lower():
            emphasis = candidate
        # else: silently dropped — not a substring, never fails the segment.

    return SegmentIntent(
        text=safe_text,
        emotion=emo,
        vocal_behavior=behavior,
        pause_after=pause,
        emphasis_word=emphasis,
        confidence=_safe_confidence(confidence),
    )


def parse_segment_intent_json(raw: str, fallback_text: str = "") -> SegmentIntent:
    """
    Alternate entry point kept for testability / non-streaming callers:
    parse one JSON object describing a segment (the audit's originally
    proposed JSON-Lines shape), e.g.
    `{"text": "...", "emotion": "warm", "vocal_behavior": "none",
      "pause_after": "breath", "emphasis_word": "tomorrow"}`.

    Not used by the live streaming transports (see module docstring for
    why the inline bracket tag is used there instead) but exercises the
    exact same validation path as `build_segment_intent`, and is the
    simplest way to unit test malformed-JSON / unknown-enum handling.

    Never raises: malformed JSON, a non-dict payload, or any decode error
    falls back to plain passthrough of `fallback_text` with neutral
    defaults.
    """
    try:
        data = json.loads(raw) if raw else None
    except (json.JSONDecodeError, TypeError, ValueError):
        data = None

    if not isinstance(data, dict):
        return build_segment_intent(fallback_text)

    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        text = fallback_text

    return build_segment_intent(
        text,
        emotion=data.get("emotion"),
        vocal_behavior=data.get("vocal_behavior"),
        pause_after=data.get("pause_after"),
        emphasis_word=data.get("emphasis_word"),
        confidence=data.get("confidence"),
    )


# ── Inline bracket-tag wire format (used by the live streaming transports) ──

_DELIVERY_TAG_RE = re.compile(r"\[DELIVERY\b([^\]]*)\]", re.IGNORECASE)
_ATTR_RE = re.compile(r"""(\w+)\s*=\s*("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|[^\s\]]+)""")
# A DELIVERY tag opener with no closing "]" anywhere after it, running to
# end-of-string — i.e. truly unclosed (a "]" anywhere later would make
# _DELIVERY_TAG_RE above match instead). Used only for final, end-of-stream
# cleanup (strip_delivery_tags): once the LLM stream has ended, any such
# fragment can only be a truncated/never-closed tag, never real spoken
# content still to come.
_DANGLING_DELIVERY_TAG_RE = re.compile(r"\[DELIVERY\b[^\]]*$", re.IGNORECASE)


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _parse_tag_attrs(raw_attrs: str) -> dict:
    attrs: dict = {}
    for attr_match in _ATTR_RE.finditer(raw_attrs or ""):
        attrs[attr_match.group(1).lower()] = _unquote(attr_match.group(2))
    return attrs


def extract_pending_delivery_tag(buf: str) -> Tuple[Optional[dict], str]:
    """
    Drain EVERY complete `[DELIVERY ...]` tag currently in `buf` and strip
    them all out, keeping the attrs of the LAST (most recent) one found —
    the tag closest to the upcoming, not-yet-flushed spoken text, matching
    the wire format's "tag immediately precedes the segment it describes"
    convention.

    Looping here (rather than a single `.search()`) matters when more than
    one complete tag lands in the SAME streamed delta (e.g. a short first
    clause that doesn't cross the flush-word-minimum, or provider batching
    during reconnect catch-up) — a single-match extraction would leave the
    second tag's literal brackets in the buffer, where they could later be
    flushed and spoken verbatim.

    Returns `(attrs_dict, buf_with_ALL_complete_tags_removed)` when at
    least one complete tag was found (`attrs_dict` maps lowercased
    attribute names to raw string values — still unvalidated at this
    point), or `(None, buf)` UNCHANGED when no complete tag is present yet,
    so a tag split across multiple streamed tokens can keep accumulating
    instead of being prematurely (and incorrectly) treated as absent.

    Never raises.
    """
    try:
        if not buf or "[DELIVERY" not in buf.upper():
            return None, buf
        attrs: Optional[dict] = None
        cleaned = buf
        while True:
            match = _DELIVERY_TAG_RE.search(cleaned)
            if not match:
                break
            attrs = _parse_tag_attrs(match.group(1))
            cleaned = cleaned[: match.start()] + cleaned[match.end() :]
        return attrs, cleaned
    except Exception:
        return None, buf


def strip_delivery_tags(text: str) -> str:
    """
    Remove EVERY `[DELIVERY ...]` tag from `text` — complete ones (via a
    global `.sub()`, which already replaces every non-overlapping match, not
    just the first) AND a trailing, never-closed/truncated one (e.g. cut off
    by `max_tokens`, or a reconnect/error ending the stream mid-tag), so a
    dangling `[DELIVERY emotion=warm behavior=` fragment can never survive
    into spoken text, the transcript, or token-budget estimation either.

    Used both for the full accumulated raw LLM output (a separate,
    simpler all-at-once cleanup vs. the incremental single-tag-at-a-time
    extraction `extract_pending_delivery_tag`/`consume_delivery_tag` do for
    live TTS streaming) AND — as a defense-in-depth safety net — on every
    flushed TTS segment in both transports, so even if a caller's flush
    boundary happens to land inside an as-yet-unrecognized tag fragment,
    the literal brackets never reach synthesis. Never raises.
    """
    if not text or "[DELIVERY" not in text.upper():
        return text
    try:
        cleaned = _DELIVERY_TAG_RE.sub("", text)
        cleaned = _DANGLING_DELIVERY_TAG_RE.sub("", cleaned)
        return cleaned
    except Exception:
        return text


def consume_delivery_tag(buf: str) -> Tuple[Optional[dict], str]:
    """
    Transport-facing alias of `extract_pending_delivery_tag` — the ONE
    shared call both `bidirectional_stream.py` and
    `conversation_orchestrator.py` use so the parsing logic is never
    duplicated between transports.
    """
    return extract_pending_delivery_tag(buf)


def segment_intent_from_tag_attrs(text: str, attrs: Optional[dict]) -> SegmentIntent:
    """
    The ONE shared translation from a raw `[DELIVERY ...]` tag's attribute
    dict (as returned by `consume_delivery_tag` — keys `emotion`/`behavior`/
    `pause`/`emphasis`/`confidence`, matching the wire-format attribute
    names the LLM emits) into a validated `SegmentIntent` for `text`. Both
    transports call this instead of re-deriving the attr-name mapping
    themselves. `attrs=None` (no pending tag for this segment) is a valid,
    common input — returns fully-neutral defaults for `text`.
    """
    attrs = attrs or {}
    return build_segment_intent(
        text,
        emotion=attrs.get("emotion"),
        vocal_behavior=attrs.get("behavior"),
        pause_after=attrs.get("pause"),
        emphasis_word=attrs.get("emphasis"),
        confidence=attrs.get("confidence"),
    )


def build_delivery_prompt_block(enabled: Optional[bool] = None) -> str:
    """
    Compact `# DELIVERY` system-prompt instruction block, shared by both
    transports' prompt builders (imported, never copy-pasted). Returns ""
    when the feature flag is off, so a disabled deployment adds zero extra
    prompt tokens and the LLM is never asked to emit the tag at all.

    Deliberately semantic-only: the LLM only ever sees the enum names
    above, never milliseconds, frame counts, or provider-specific tag
    strings like "[chuckles]" — realization into those is entirely the
    deterministic policy/provider layer's job
    (`app.voice.humanization_engine`, `app.voice.tts_provider_capabilities`).
    """
    if enabled is None:
        enabled = bool(getattr(settings, "VOICE_ENABLE_LLM_HUMANIZATION", False))
    if not enabled:
        return ""

    emotions = "|".join(e.value for e in DeliveryEmotion)
    behaviors = "|".join(b.value for b in VocalBehavior)
    pauses = "|".join(p.value for p in PauseCategory)
    return (
        "# DELIVERY\n"
        "Immediately before each sentence or short clause you speak, you MAY "
        f"prefix it with one tag: [DELIVERY emotion={emotions} "
        f"behavior={behaviors} pause={pauses}]\n"
        "- Defaults (emotion=neutral behavior=none pause=none) are correct "
        "for most segments — only deviate when it genuinely fits the "
        "content and the caller's mood.\n"
        "- behavior is rare: use soft_chuckle/brief_sigh/hesitation at most "
        "once every few turns, never twice in a row.\n"
        '- Optionally add emphasis="word" with exactly one word taken '
        "verbatim from the sentence that follows, to lightly stress it.\n"
        "- The tag is removed before speech and the caller never hears it — "
        "never mention, explain, or spell it out, and never let it replace "
        "any word the caller needs to hear.\n"
        "- If unsure, omit the tag entirely and just speak normally."
    )
