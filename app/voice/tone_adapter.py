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
    r"^(?:Awesome!|Great!|That'?s (?:great|wonderful)!|Perfect!|Super!|Excellent!)\s*",
    re.I,
)


def tone_adapter(text: str, ctx: "TurnContext", use_ssml: bool) -> str:
    """
    Light-touch rewrites for spoken output. SSML is mostly unchanged except leading chipper lines.
    """
    if not text or not (text := text.strip()):
        return text

    mood = ctx.mood

    if not use_ssml:
        t = _LEADING_CHIPPER.sub("", text)
        if t != text:
            text = t.strip() or text

    if mood in (UserMood.SAD, UserMood.FRUSTRATED, UserMood.ANGRY, UserMood.URGENT):
        if not use_ssml:
            for a, b in (
                (r"\bNo worries!\b", "No worries."),
                (r"\bNo problem!\b", "No problem."),
                (r"\bSounds good!\b", "Sounds good."),
                (r"\bLove to help!\b", "Happy to help."),
            ):
                text = re.sub(a, b, text, flags=re.I)
    if mood in (UserMood.HAPPY, UserMood.NEUTRAL) and not use_ssml:
        return text
    if mood == UserMood.SAD and not use_ssml:
        text = re.sub(r"\b(Yay!|Woohoo!)\b", "Okay.", text, flags=re.I)

    return text
