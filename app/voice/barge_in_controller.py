"""
BargeInController: Audio-frame-level interruption detection and safe cancellation.

Monitors interim transcripts + agent_speaking flag. On barge-in, triggers instant
cancellation cascade via VoiceOrchestrator.on_barge_in().

Design: Sub-100ms interrupt — operates at transcript level, NOT queue level.
"""

import logging
from typing import TYPE_CHECKING

from app.core.config import settings

if TYPE_CHECKING:
    from app.voice.orchestrator import VoiceOrchestrator

logger = logging.getLogger(__name__)


class BargeInController:
    """
    Detects user interruption of agent speech and triggers cancellation cascade.

    Thresholds (from settings, tunable per-agent):
      - multi_word: 2+ words @ 0.25 confidence → barge-in
      - single_word: "stop"/"no"/"halt" @ 0.52 confidence → immediate barge-in

    IMPORTANT: trigger_barge_in() is idempotent — safe to call multiple times per turn.
    """

    # Words that trigger single-word barge-in at high confidence
    STOP_WORDS = frozenset({"stop", "no", "halt", "wait", "enough", "quiet", "shush"})

    def __init__(self, call_id: str, orchestrator: "VoiceOrchestrator") -> None:
        self.call_id = call_id
        self.orchestrator = orchestrator

        # Configurable thresholds (can be overridden per-agent)
        self.multi_word_threshold: float = settings.VOICE_BARGE_IN_MIN_CONFIDENCE
        self.single_word_threshold: float = settings.VOICE_BARGE_IN_MIN_CONFIDENCE_1W

        # Guard: prevent double-triggering within a single agent turn
        self._barge_in_triggered: bool = False

    def arm(self) -> None:
        """
        Re-arm barge-in detection for a new agent utterance.

        Must be called each time the agent starts speaking so that the next
        user interim can trigger cancellation again.
        """
        self._barge_in_triggered = False
        logger.debug(f"[{self.call_id}] BargeInController armed")

    def disarm(self) -> None:
        """
        Disarm barge-in detection (e.g., agent has finished speaking).

        Prevents stale interims from triggering barge-in after TTS completes.
        """
        self._barge_in_triggered = False
        logger.debug(f"[{self.call_id}] BargeInController disarmed")

    def configure_thresholds(
        self,
        multi_word_threshold: float,
        single_word_threshold: float,
    ) -> None:
        """
        Runtime threshold adjustment (per-agent or per-call tuning).

        Args:
            multi_word_threshold: Confidence threshold for 2+ word interrupts.
            single_word_threshold: Confidence threshold for single-word stop-words.
        """
        self.multi_word_threshold = multi_word_threshold
        self.single_word_threshold = single_word_threshold
        logger.debug(
            f"[{self.call_id}] BargeIn thresholds updated: "
            f"multi={multi_word_threshold}, single={single_word_threshold}"
        )

    async def check_trigger(self, interim_text: str, confidence: float) -> None:
        """
        Evaluate interim transcript for barge-in conditions.

        Called from VoiceOrchestrator.on_stt_interim() ONLY when agent is speaking.

        Args:
            interim_text: Latest interim transcript from Deepgram.
            confidence: STT confidence (0.0–1.0).
        """
        # Already triggered this turn — don't cascade again
        if self._barge_in_triggered:
            return

        words = interim_text.strip().split()
        word_count = len(words)

        if word_count == 0:
            return

        # --- Multi-word barge-in (primary path) ---
        if word_count >= 2 and confidence >= self.multi_word_threshold:
            logger.info(
                f"[{self.call_id}] Barge-in: multi-word '{interim_text}' "
                f"({word_count}w @ {confidence:.2f} ≥ {self.multi_word_threshold})"
            )
            await self._trigger_barge_in()
            return

        # --- Single-word stop-word barge-in (emergency path) ---
        if (
            word_count == 1
            and words[0].lower() in self.STOP_WORDS
            and confidence >= self.single_word_threshold
        ):
            logger.info(
                f"[{self.call_id}] Barge-in: stop-word '{words[0]}' "
                f"@ {confidence:.2f} ≥ {self.single_word_threshold}"
            )
            await self._trigger_barge_in()
            return

    async def _trigger_barge_in(self) -> None:
        """
        Internal: execute barge-in cascade once per turn.

        Sets guard flag first to prevent re-entrancy, then notifies orchestrator.
        The orchestrator cancels all running tasks (LLM + TTS) via CancellationToken.
        """
        self._barge_in_triggered = True
        await self.orchestrator.on_barge_in()
