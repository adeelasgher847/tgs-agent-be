"""
Local, O(1) user-turn signals for voice: mood heuristics and prompt steering.

No extra LLM or network calls — keeps the bidirectional STT → LLM → TTS path flat.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class UserMood(str, Enum):
    """Coarse mood bucket for system prompt and TTS post-processing."""

    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    URGENT = "urgent"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"


@dataclass
class TurnContext:
    """Structured signals for one user utterance (STT final or interim)."""

    mood: UserMood
    respond_briefly: bool
    conversation_phase: str
    stt_confidence: float
    # Optional 0.0–1.0 for future TTS prosody mapping; no extra API in this path.
    tts_stability_hint: Optional[float] = None
    is_final: bool = True

    def mood_label(self) -> str:
        return self.mood.value


# Lower = checked first; order matters for overlapping keywords.
_URGENT = re.compile(
    r"\b(urgent|emergency|asap|immediately|right now|help me|sos)\b", re.I
)
_FRUSTRATED = re.compile(
    r"\b(frustrat|annoy|ridiculous|waste|not working|sick of|tired of|terrible|useless|worst|awful|horrible|disgust|useless|cancel)\b",
    re.I,
)
_ANGRY = re.compile(
    r"\b(damn|furious|fuck|hate|angry|mad at|outrage|unacceptable|dispute|lawsuit|sue|refund|complain)\b",
    re.I,
)
_SAD = re.compile(
    r"\b(sad|unfortunately|pass(ed)? away|died|funeral|can'?t (afford|cope|take)|overwhelmed|depress|grief|lost my|sorry for myself)\b",
    re.I,
)
_HAPPY = re.compile(
    r"\b(thank(s| you)|appreciate|great|wonderful|love it|awesome|excited|perfect|that helps)\b", re.I,
)


def detect_mood(user_text: str, stt_confidence: float = 0.0) -> UserMood:
    """
    Heuristic mood from text + weak confidence weighting.
    Does not call external services.
    """
    text = (user_text or "").strip()
    if not text:
        return UserMood.NEUTRAL

    t = text.lower()
    if _URGENT.search(t) and stt_confidence >= 0.2:
        return UserMood.URGENT
    if _ANGRY.search(t):
        return UserMood.ANGRY
    if _FRUSTRATED.search(t):
        return UserMood.FRUSTRATED
    if _SAD.search(t):
        return UserMood.SAD
    if _HAPPY.search(t) and stt_confidence >= 0.15:
        return UserMood.HAPPY
    if "!" in text and len(t.split()) <= 5 and stt_confidence < 0.4:
        return UserMood.FRUSTRATED
    return UserMood.NEUTRAL


def _respond_briefly(user_text: str, mood: UserMood, stt_confidence: float) -> bool:
    words = len((user_text or "").split())
    if words <= 3:
        return True
    if mood in (UserMood.URGENT, UserMood.ANGRY) and words < 20:
        return True
    if stt_confidence < 0.35 and words < 8:
        return True
    return False


def _tts_stability_for_mood(mood: UserMood) -> Optional[float]:
    """Slightly lower = more variable prosody; higher = calmer. For future ElevenLabs mapping."""
    if mood in (UserMood.ANGRY, UserMood.FRUSTRATED, UserMood.URGENT):
        return 0.42
    if mood in (UserMood.SAD,):
        return 0.55
    if mood in (UserMood.HAPPY,):
        return 0.50
    return 0.48


def build_turn_context(
    user_text: str,
    stt_confidence: float = 0.0,
    *,
    booking_context_active: bool = False,
    is_final: bool = True,
) -> TurnContext:
    mood = detect_mood(user_text, stt_confidence)
    if booking_context_active:
        phase = "booking"
    else:
        phase = "general"
    return TurnContext(
        mood=mood,
        respond_briefly=_respond_briefly(user_text, mood, stt_confidence),
        conversation_phase=phase,
        stt_confidence=float(stt_confidence or 0.0),
        tts_stability_hint=_tts_stability_for_mood(mood),
        is_final=is_final,
    )


def build_user_signals_block(ctx: TurnContext) -> str:
    """
    Short block injected into the LLM system prompt (single call, no extra latency).
    """
    mood = ctx.mood_label()
    brief = "yes" if ctx.respond_briefly else "no"
    return (
        f"# USER_SIGNALS (heuristic, do not mention explicitly)\n"
        f"- inferred_mood: {mood}\n"
        f"- respond_briefly: {brief}\n"
        f"- conversation_phase: {ctx.conversation_phase}\n"
        f"- stt_confidence: {ctx.stt_confidence:.2f}\n"
        f"- When inferred_mood is frustrated, angry, or urgent: acknowledge and help first; be concise; avoid cheerful small talk.\n"
        f"- When inferred_mood is sad: be warm, gentle, and patient; do not be overly enthusiastic.\n"
        f"- When respond_briefly is yes: prefer very short TTS-friendly sentences."
    )
