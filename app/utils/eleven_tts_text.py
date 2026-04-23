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
_CONTROL_TOKEN_RE = re.compile(r"\[(?:END_CALL|OUTCOME:|CHECK_SLOTS:|BOOK_APPOINTMENT:)", re.IGNORECASE)

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
        return apply_elevenlabs_breathing_fallback(text)
    return strip_eleven_v3_style_tags_for_non_eleven_tts(text)


def supports_elevenlabs_audio_tags(provider_slug: Optional[str]) -> bool:
    """Return True only for ElevenLabs TTS, where bracketed audio tags are valid."""
    return (provider_slug or "").lower() == "elevenlabs"


def contains_elevenlabs_audio_tag(text: str) -> bool:
    """True when text already includes a known Eleven-style bracketed audio tag."""
    if not text or "[" not in text:
        return False
    for match in _TAG_RE.finditer(text):
        if _normalize_tag_inner(match.group(1) or "") in _ELEVEN_V3_TAG_INNERS:
            return True
    return False


def apply_elevenlabs_breathing_fallback(text: str) -> str:
    """
    Add a subtle leading [breathes] tag for longer ElevenLabs utterances when the
    model did not emit any known audio tag itself.

    Safety rules:
    - Never touch control-token responses.
    - Never add a duplicate if a known audio tag already exists.
    - Skip very short transactional replies.
    """
    if not text or not text.strip():
        return text
    if _CONTROL_TOKEN_RE.search(text):
        return text
    if contains_elevenlabs_audio_tag(text):
        return text

    stripped = text.strip()
    word_count = len(stripped.split())
    has_long_form_shape = word_count >= 9 or any(mark in stripped for mark in (".", "?", "!", ",", ";", ":"))
    if word_count < 6 or not has_long_form_shape:
        return stripped
    return f"[breathes] {stripped}"


def build_elevenlabs_audio_tag_prompt_block(provider_slug: Optional[str]) -> str:
    """
    Guidance injected into voice prompts.
    Only enable bracketed audio tags for ElevenLabs, where the TTS engine can
    interpret them as non-verbal cues instead of speaking them literally.
    """
    if not supports_elevenlabs_audio_tags(provider_slug):
        return ""
    return (
        "# ELEVENLABS AUDIO TAGS\n"
        "- You may use a sparse bracketed audio tag like [breathes] when it would sound natural in speech.\n"
        "- Use at most one such tag in a normal reply, and skip it for short transactional replies.\n"
        "- Never put audio tags inside system/control tokens like [CHECK_SLOTS:...], "
        "[BOOK_APPOINTMENT:...], or [END_CALL].\n"
    )
