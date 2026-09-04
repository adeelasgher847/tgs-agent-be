"""
ElevenLabs v3 (and similar) use bracketed *audio tags* in TTS text, e.g. [breathes] [pause] [excited] [sad].
Those must not reach other providers: Google TTS can speak the brackets and/or mis-handle SSML.

Under the V-08 humanization architecture, the LLM never emits these tags itself — it only
emits the semantic `[DELIVERY ...]` tag (see `app/voice/humanization_intent.py`). Real
provider-specific bracketed tags (e.g. ElevenLabs `[chuckles]`/`[sighs]`) are added
programmatically after the fact by `apply_vocal_behavior_tag()` in
`app/voice/tts_provider_capabilities.py`, which reuses `_ELEVEN_V3_TAG_INNERS` below.

This module:
- For provider **elevenlabs**: returns text unchanged (no extra work when no [] present).
- For any other provider: removes only **known** tag inners; unknown `[...]` is left as-is
  to avoid deleting user content like [SKU-100] (digits help distinguish later if needed).
"""

from __future__ import annotations

import re

from app.utils.ssml_utils import strip_ssml_tags


# Single-pass regex; only substitute when inner normalizes to a known tag
_TAG_RE = re.compile(r"\[([^\]]*)\]")
_CONTROL_TOKEN_RE = re.compile(
    r"\[(?:END_CALL|TRANSFER_CALL|SCREENING_QUALIFIED|OUTCOME:|CHECK_SLOTS:|BOOK_APPOINTMENT:)",
    re.IGNORECASE,
)
# Strip literal pause tags for ElevenLabs so models don't speak "pause".
_PAUSE_TAG_RE = re.compile(r"\[\s*(?:pause|pauses)\s*\]", re.IGNORECASE)

# Normalized: whitespace collapsed, lowercased. Expand as Eleven documents new tags.
_ELEVEN_V3_TAG_INNERS: frozenset[str] = frozenset(
    {
        "breathes",
        "breathe",
        "breath",
        "breathes heavily",
        "heavy breathing",
        "breathe in",
        "breathe out",
        "sigh",
        "sighs",
        "deep sigh",
        "sighs deeply",
        "sigh of relief",
        "whispers",
        "whisper",
        "whispering",
        "shouts",
        "shout",
        "laughs",
        "laugh",
        "laughing",
        "giggles",
        "chuckles",
        "laughs softly",
        "nervous laugh",
        "nervous laughter",
        "gasps",
        "gasp",
        "gulps",
        "gulp",
        "clears throat",
        "coughs",
        "cough",
        "sniffles",
        "pauses",
        "pause",
        "stammers",
        "stutter",
        "hesitates",
        "tired",
        "nervous",
        "calm",
        "excited",
        "sad",
        "sadly",
        "sorrowful",
        "nervously",
    }
)


def _normalize_tag_inner(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def strip_eleven_v3_style_tags_for_non_eleven_tts(text: str) -> str:
    """
    Remove only whitelisted [tag] segments. O(n) per call; one regex scan; no I/O.
    If text has no '[', returns immediately.
    """
    if not text or "[" not in text:
        return text

    def _repl(m: re.Match) -> str:
        raw = m.group(1)
        if not raw.strip():
            return ""
        key = _normalize_tag_inner(raw)
        if key in _ELEVEN_V3_TAG_INNERS:
            return ""
        return m.group(0)

    out = _TAG_RE.sub(_repl, text)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


def prepare_tts_text_for_provider(text: str, provider_slug: str | None) -> str:
    """
    Sanitize text before it reaches any TTS provider adapter.
    - Strips all complete and incomplete SSML/XML tags (<speak>, <prosody>, <break>, etc.).
    - Strips all bracket audio tags ([breathes], [excited], [sad], [pause], etc.).
    - Strips backend control tokens ([END_CALL], [TRANSFER_CALL], [SCREENING_QUALIFIED], etc.).
    - Preserves legitimate business bracket text (e.g. [SKU-100], [Order #123]).
    """
    if not text:
        return ""

    # 1. Defensively strip all SSML / XML tags
    cleaned = strip_ssml_tags(text)
    if not cleaned:
        return ""

    if "[" not in cleaned:
        return cleaned

    # 2. Strip pause tags and control tokens
    cleaned = _PAUSE_TAG_RE.sub("", cleaned)
    cleaned = _CONTROL_TOKEN_RE.sub("", cleaned)

    # 3. Strip all known bracketed prosody/audio tags for all providers
    def _repl_audio_tag(m: re.Match) -> str:
        raw = m.group(1)
        if not raw.strip():
            return ""
        key = _normalize_tag_inner(raw)
        if key in _ELEVEN_V3_TAG_INNERS:
            return ""
        return m.group(0)

    cleaned = _TAG_RE.sub(_repl_audio_tag, cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


