"""
CPU-only TTS text shaping from TurnContext. Runs on LLM output chunks before queue_tts.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.voice.turn_signals import TurnContext

from app.voice.turn_signals import UserMood


_LEADING_CHIPPER = re.compile(
    r"^(?:Awesome!|Great!|Fantastic!|Wonderful!|That'?s (?:great|wonderful)!|"
    r"Perfect!|Super!|Excellent!)\s*",
    re.I,
)

_SSML_MARKUP = re.compile(r"<\s*(?:speak|prosody|break|emphasis|say-as)\b", re.I)

# Spoken substitutions that never touch names, numbers, or URLs.
_FORMAL_SPOKEN = (
    (re.compile(r"\bHow may I assist you\b", re.I), "How can I help"),
    (re.compile(r"\bHow may I help you\b", re.I), "How can I help"),
    (re.compile(r"\bI would be happy to\b", re.I), "I'd be happy to"),
    (re.compile(r"\bI apologize for any inconvenience\b", re.I), "Sorry about that"),
    (re.compile(r"\bI apologize for the inconvenience\b", re.I), "Sorry about that"),
    (re.compile(r"\bPlease be advised that\s*", re.I), ""),
    (re.compile(r"\bPlease be advised\s*", re.I), ""),
    (re.compile(r"\bAs an AI(?: language model)?[,]?\s*", re.I), ""),
    (re.compile(r"^Certainly[!.]?\s+", re.I), "Sure. "),
)

_CONTRACTIONS = (
    (re.compile(r"\bI am\b"), "I'm"),
    (re.compile(r"\bI will\b"), "I'll"),
    (re.compile(r"\bI would\b"), "I'd"),
    (re.compile(r"\bcannot\b", re.I), "can't"),
    (re.compile(r"\bdo not\b", re.I), "don't"),
    (re.compile(r"\bdoes not\b", re.I), "doesn't"),
    (re.compile(r"\bwe will\b", re.I), "we'll"),
)

_COLLAPSE_SPACE = re.compile(r"[ \t]{2,}")


def _looks_like_ssml(text: str) -> bool:
    return bool(_SSML_MARKUP.search(text))


def tone_adapter(text: str, ctx: "TurnContext", use_ssml: bool) -> str:
    """
    Light-touch rewrites for spoken output.

    Plain-text shaping still runs when ``use_ssml`` is True: production handlers
    wrap TTS in SSML *after* this step. Skip only when the chunk already contains
    SSML markup so tags are not rewritten.
    """
    if not text or not (text := text.strip()):
        return text

    # use_ssml is kept for call-site parity; production wraps SSML after this.
    _ = use_ssml
    if _looks_like_ssml(text):
        return text

    mood = ctx.mood

    t = _LEADING_CHIPPER.sub("", text)
    if t != text:
        text = t.strip() or text

    for pattern, repl in _FORMAL_SPOKEN:
        text = pattern.sub(repl, text)
    for pattern, repl in _CONTRACTIONS:
        text = pattern.sub(repl, text)
    text = _COLLAPSE_SPACE.sub(" ", text).strip()

    if mood in (UserMood.SAD, UserMood.FRUSTRATED, UserMood.ANGRY, UserMood.URGENT):
        for a, b in (
            (r"\bNo worries!", "No worries."),
            (r"\bNo problem!", "No problem."),
            (r"\bSounds good!", "Sounds good."),
            (r"\bLove to help!", "Happy to help."),
        ):
            text = re.sub(a, b, text, flags=re.I)
    if mood == UserMood.SAD:
        text = re.sub(r"\b(Yay!|Woohoo!)", "Okay.", text, flags=re.I)

    return text
