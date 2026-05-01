"""
Lightweight latency markers for voice (STT → LLM → TTS). Observability only.

All timestamps use time.perf_counter() for monotonic deltas. No I/O in hot paths.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class VoiceTurnMetrics:
    """Per-call metrics with one active user turn at a time."""

    call_pickup_mono: Optional[float] = None
    turn_stt_final_mono: Optional[float] = None
    turn_llm_first_token_mono: Optional[float] = None
    turn_first_tts_queued_mono: Optional[float] = None

    def mark_call_pickup(self) -> None:
        if self.call_pickup_mono is None:
            self.call_pickup_mono = time.perf_counter()

    def begin_turn_at_stt_final(self) -> None:
        """Call once per accepted final STT before LLM work runs."""
        now = time.perf_counter()
        self.turn_stt_final_mono = now
        self.turn_llm_first_token_mono = None
        self.turn_first_tts_queued_mono = None

    def mark_llm_first_token(self) -> None:
        if self.turn_llm_first_token_mono is None:
            self.turn_llm_first_token_mono = time.perf_counter()

    def mark_first_tts_queued(self) -> None:
        if self.turn_first_tts_queued_mono is None:
            self.turn_first_tts_queued_mono = time.perf_counter()

    def log_turn_summary(
        self,
        log: Any,
        *,
        user_preview: str = "",
        session_hint: str = "",
    ) -> None:
        """Best-effort INFO log with sub-second deltas (missing legs omitted)."""
        if self.turn_stt_final_mono is None:
            return
        t0 = self.turn_stt_final_mono
        parts = [f"stt_final→now={time.perf_counter() - t0:.3f}s"]
        if self.turn_llm_first_token_mono is not None:
            parts.append(
                f"stt_final→llm_1st={self.turn_llm_first_token_mono - t0:.3f}s"
            )
        if self.turn_first_tts_queued_mono is not None:
            parts.append(
                f"stt_final→tts_q_1st={self.turn_first_tts_queued_mono - t0:.3f}s"
            )
        suf = f" session={session_hint}" if session_hint else ""
        prev = (user_preview[:48] + "…") if len(user_preview) > 48 else user_preview
        log.info("[VoiceMetrics] turn_latency %s user=%r%s", " ".join(parts), prev, suf)
