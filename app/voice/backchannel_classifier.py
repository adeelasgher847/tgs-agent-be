"""
Backchannel and Turn-State Classifier for conversational voice streams.

Distinguishes between:
1. BARGE_IN: Explicit interruption commands or actionable user turns spoken over active TTS.
2. SUPPRESS_NON_ACTIONABLE_BACKCHANNEL: Known non-actionable conversational backchannels,
   fillers, and brief greetings spoken while the agent is speaking.
3. NORMAL_USER_TURN: Genuine user requests to be processed normally.

Design Rules:
- Unknown/unseen phrases are NEVER assumed to be backchannels.
- Only confidently matched non-actionable backchannels/greetings are suppressed during active TTS.
- Explicit interruption commands always barge in.
- When TTS is not active, all genuine user utterances are treated as NORMAL_USER_TURN.
"""

from __future__ import annotations

import re
from enum import Enum


class TurnClassification(str, Enum):
    BARGE_IN = "barge_in"
    SUPPRESS_NON_ACTIONABLE_BACKCHANNEL = "suppress_non_actionable_backchannel"
    NORMAL_USER_TURN = "normal_user_turn"


# Pure acoustic fillers (single or multi-token repetitions)
ACOUSTIC_FILLERS = frozenset(
    {
        "uh",
        "um",
        "hmm",
        "mm",
        "ah",
        "er",
        "huh",
        "mhm",
        "mmm",
        "eh",
        "ugh",
    }
)

# Known, non-actionable conversational backchannels, affirmations, and greetings
# that callers routinely utter while listening to the agent speak.
KNOWN_BACKCHANNEL_PHRASES = frozenset(
    {
        # Greetings & pleasantries
        "hey",
        "hi",
        "hello",
        "hey hi",
        "hi hey",
        "hi there",
        "hello there",
        "hey there",
        "good morning",
        "good afternoon",
        "good evening",
        "good day",
        # Affirmations & acknowledgements
        "yeah",
        "yes",
        "yep",
        "yup",
        "yeah yeah",
        "yes yes",
        "yeah sure",
        "yes sure",
        "okay",
        "ok",
        "okay okay",
        "ok ok",
        "okay and",
        "ok and",
        "okay yeah",
        "ok yeah",
        "right",
        "right right",
        "alright",
        "all right",
        "alright alright",
        "sure",
        "sure thing",
        "got it",
        "i see",
        "i got it",
        "understood",
        "makes sense",
        "thank you",
        "thanks",
        "thanks a lot",
        "thank you very much",
        # Multi-word filler pairs
        "uh huh",
        "mhm mhm",
        "hmm hmm",
        "ah yes",
        "oh okay",
        "oh ok",
        "oh yeah",
        "oh i see",
        "oh got it",
    }
)

# Explicit command keywords and stems that indicate the caller wants the agent to stop/pause/cancel
EXPLICIT_COMMAND_KEYWORDS = frozenset(
    {
        "stop",
        "wait",
        "hold",
        "pause",
        "cancel",
        "interrupt",
        "shut",
        "quiet",
        "listen",
        "silence",
        "shh",
        "abort",
        "quit",
    }
)

# Explicit multi-word command phrases
EXPLICIT_COMMAND_PHRASES = frozenset(
    {
        "stop please",
        "please stop",
        "stop talking",
        "stop speaking",
        "wait please",
        "please wait",
        "wait a second",
        "wait a minute",
        "wait one second",
        "wait one moment",
        "just wait",
        "hold on",
        "hold up",
        "please hold",
        "hold on please",
        "pause please",
        "please pause",
        "cancel that",
        "cancel please",
        "please cancel",
        "shut up",
        "be quiet",
        "quiet please",
        "listen to me",
        "listen please",
        "hang up",
    }
)


def normalize_transcript(text: str) -> str:
    """Lowercase and strip non-alphanumeric characters, returning single-space separated words."""
    if not text:
        return ""
    no_apostrophe = re.sub(r"['’]", "", text)
    cleaned = re.sub(r"[^a-zA-Z0-9\s]+", " ", no_apostrophe).lower()
    return " ".join(cleaned.split())


def is_pure_acoustic_filler(text: str) -> bool:
    """Check if the text consists entirely of acoustic fillers (uh, um, mhm, uh huh)."""
    norm = normalize_transcript(text)
    if not norm:
        return True
    tokens = norm.split()
    return bool(tokens) and all(t in ACOUSTIC_FILLERS for t in tokens)


def is_known_non_actionable_backchannel(text: str) -> bool:
    """
    Conservative classifier: returns True ONLY if the utterance confidently matches
    a known non-actionable conversational backchannel/greeting.

    Unknown or ambiguous phrases return False so they are NEVER silently lost.
    """
    norm = normalize_transcript(text)
    if not norm:
        return True
    if is_pure_acoustic_filler(text):
        return True
    if norm in KNOWN_BACKCHANNEL_PHRASES:
        return True
    return False


def has_explicit_interruption_intent(text: str) -> bool:
    """Check if the utterance contains an explicit interruption or command keyword/phrase."""
    norm = normalize_transcript(text)
    if not norm:
        return False
    if norm in EXPLICIT_COMMAND_PHRASES:
        return True
    tokens = norm.split()
    return any(t in EXPLICIT_COMMAND_KEYWORDS for t in tokens)


def classify_turn(
    transcript: str,
    confidence: float,
    *,
    is_tts_playing: bool,
    min_confidence: float = 0.26,
    min_words: int = 2,
    min_confidence_1w: float = 0.36,
) -> TurnClassification:
    """
    Classify an STT event into one of three states:
    1. BARGE_IN: Should interrupt active TTS playback.
    2. SUPPRESS_NON_ACTIONABLE_BACKCHANNEL: Should be suppressed while TTS is playing (no interruption, no LLM).
    3. NORMAL_USER_TURN: Genuine user turn that should be processed by the conversation pipeline.
    """
    norm = normalize_transcript(transcript)
    if not norm:
        return TurnClassification.SUPPRESS_NON_ACTIONABLE_BACKCHANNEL

    # Pure acoustic fillers are always suppressed regardless of TTS state
    if is_pure_acoustic_filler(norm):
        return TurnClassification.SUPPRESS_NON_ACTIONABLE_BACKCHANNEL

    # When TTS is NOT active, everything non-filler is a normal user turn
    if not is_tts_playing:
        return TurnClassification.NORMAL_USER_TURN

    # ── Active TTS is playing ──

    # 1. Explicit interruption commands always barge in (even if 1-2 words, e.g. "stop", "hold on")
    if has_explicit_interruption_intent(norm):
        req_conf = min_confidence_1w if len(norm.split()) == 1 else min_confidence
        if confidence >= req_conf:
            return TurnClassification.BARGE_IN
        return TurnClassification.SUPPRESS_NON_ACTIONABLE_BACKCHANNEL

    # 2. Known conversational backchannels & greetings are suppressed while TTS is active
    if is_known_non_actionable_backchannel(norm):
        return TurnClassification.SUPPRESS_NON_ACTIONABLE_BACKCHANNEL

    # 3. Legitimate user requests & unknown phrases spoken over TTS
    # If confidence meets threshold and word count is sufficient, treat as barge-in to capture the request
    tokens = norm.split()
    word_count = len(tokens)

    if word_count >= min_words:
        if confidence >= min_confidence:
            return TurnClassification.BARGE_IN
        return TurnClassification.SUPPRESS_NON_ACTIONABLE_BACKCHANNEL

    # Single-word non-command, non-backchannel spoken over TTS (e.g. single unknown word below min_words)
    if min_words == 1 and confidence >= min_confidence_1w:
        return TurnClassification.BARGE_IN

    return TurnClassification.SUPPRESS_NON_ACTIONABLE_BACKCHANNEL
