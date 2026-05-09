"""
ElevenLabs v3 (and similar) use bracketed *audio tags* in TTS text, e.g. [breathes] [pause] [excited] [sad].
Those must not reach other providers: Google TTS can speak the brackets and/or mis-handle SSML.

This module:
- For provider **elevenlabs**: returns text unchanged (no extra work when no [] present).
- For any other provider: removes only **known** tag inners; unknown `[...]` is left as-is
  to avoid deleting user content like [SKU-100] (digits help distinguish later if needed).
- LLM/voice prompt guidance and tag enablement: `settings.ENABLE_ELEVENLABS_AUDIO_TAGS` plus
  `supports_elevenlabs_audio_tags` (ElevenLabs `tts_provider` slug only).
"""

from __future__ import annotations

import re
from typing import Optional

from app.core.config import settings

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


def prepare_tts_text_for_provider(text: str, provider_slug: Optional[str]) -> str:
    """
    ElevenLabs: pass through unchanged (keeps v3 audio tags in the string).
    All other TTS providers: strip known Eleven-style bracket tags.
    No network; negligible CPU; safe when text has no tags (fast path: no '[').
    """
    if (provider_slug or "").lower() == "elevenlabs":
        # Some Eleven model variants can speak [pause]/[pauses] literally as
        # the word "pause". Strip only those tags to prevent audible artifacts
        # while preserving other optional expressive tags.
        if not text or "[" not in text:
            return text
        out = _PAUSE_TAG_RE.sub("", text)
        out = re.sub(r"[ \t]{2,}", " ", out)
        return out.strip()
    return strip_eleven_v3_style_tags_for_non_eleven_tts(text)


def supports_elevenlabs_audio_tags(provider_slug: Optional[str]) -> bool:
    """
    Return True when the agent's TTS is ElevenLabs and tag guidance is enabled in settings.
    All tag-related LLM text and fallbacks should be gated on this.
    """
    if (provider_slug or "").lower() != "elevenlabs":
        return False
    return bool(getattr(settings, "ENABLE_ELEVENLABS_AUDIO_TAGS", True))


def get_elevenlabs_voice_prompt_rule_lines() -> tuple[str, str, str]:
    """
    (output_plain_text_rule, no_ssml_rule_base, no_ssml_rule) for the base / custom
    voice system prompts. Only call when supports_elevenlabs_audio_tags is True
    to avoid duplicate instruction blocks for non–ElevenLabs TTS.
    """
    # Short inline reminder; the detailed policy lives in build_elevenlabs_audio_tag_prompt_block.
    short = (
        "Optional: at most ONE ElevenLabs audio tag per reply when it clearly helps: "
        "[breathes] or [breathe], [excited], [sad] or [sorrowful] — "
        "not every line; most replies have zero tags. See # ELEVENLABS AUDIO TAGS below."
    )
    return (
        f"- OUTPUT PLAIN TEXT ONLY: Do NOT output SSML or XML. {short}",
        f"4. NO SSML: Do NOT output <speak>, <prosody>, or any XML tags. Plain text only. {short}",
        f"3. NO SSML: Plain text only. No <speak>, <prosody>, or XML. {short}",
    )


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
    Only when ElevenLabs TTS is in use and ENABLE_ELEVENLABS_AUDIO_TAGS is True; other
    providers must never receive this block (brackets can be read aloud or break TTS).
    """
    if not supports_elevenlabs_audio_tags(provider_slug):
        return ""
    return (
        "# ELEVENLABS AUDIO TAGS\n"
        "- This call uses **ElevenLabs** TTS. You *may* add **one** optional bracketed audio tag per reply, "
        "**only** when the delivery would clearly benefit. Default is **no** tag; do not add tags in every message.\n"
        "- **Allowed (pick at most one, only when needed):** "
        "[breathes] or [breathe] (light lead-in); "
        "[excited] (genuine energy); [sad] or [sorrowful] (empathy — use sparingly, professional call tone).\n"
        "- **When to skip tags:** short transactional replies (yes/no, numbers, times, one-line confirmations), "
        "or any reply where plain text is enough.\n"
        "- **Placement:** start of the spoken line, or a single short beat before a sentence—never inside a word.\n"
        "- **Never** put audio tags inside system tokens: [CHECK_SLOTS:...], [BOOK_APPOINTMENT:...], "
        "[END_CALL], [SCREENING_QUALIFIED], or [OUTCOME:...].\n"
    )
