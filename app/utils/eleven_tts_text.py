"""
ElevenLabs v3 (and similar) use bracketed *audio tags* in TTS text, e.g. [breathes] [sigh].
Those must not reach other providers: Google TTS can speak the brackets and/or mis-handle SSML.

This module:
- For provider **elevenlabs**: returns text unchanged (no extra work when no [] present).
- For any other provider: removes only **known** tag inners; unknown `[...]` is left as-is
  to avoid deleting user content like [SKU-100] (digits help distinguish later if needed).
"""

from __future__ import annotations

import re
from typing import Optional

# Single-pass regex; only substitute when inner normalizes to a known tag
_TAG_RE = re.compile(r"\[([^\]]*)\]")

# Normalized: whitespace collapsed, lowercased. Expand as Eleven documents new tags.
_ELEVEN_V3_TAG_INNERS: frozenset[str] = frozenset(
    {
        "breathes",
        "breathe",
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


def prepare_tts_text_for_provider(text: str, provider_slug: Optional[str]) -> str:
    """
    ElevenLabs: pass through unchanged (keeps v3 audio tags in the string).
    All other TTS providers: strip known Eleven-style bracket tags.
    No network; negligible CPU; safe when text has no tags (fast path: no '[').
    """
    if (provider_slug or "").lower() == "elevenlabs":
        return text
    return strip_eleven_v3_style_tags_for_non_eleven_tts(text)
