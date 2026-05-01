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
    # Anchor for each LLM+TTS generation (interim or final). Prevents stale STT timestamps
    # from inflating latency logs when speculation runs before/without a new final.
    generation_anchor_mono: Optional[float] = None
    turn_llm_first_token_mono: Optional[float] = None
    turn_first_tts_queued_mono: Optional[float] = None

    def mark_call_pickup(self) -> None:
        if self.call_pickup_mono is None:
            self.call_pickup_mono = time.perf_counter()

    def begin_turn_at_stt_final(self) -> None:
        """Record accepted final STT time (correlation only). LLM markers reset in start_generation."""
        self.turn_stt_final_mono = time.perf_counter()

    def start_generation(self) -> None:
        """Call at the start of each generate_and_stream_response (non-greeting)."""
        self.generation_anchor_mono = time.perf_counter()
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
        t0 = self.generation_anchor_mono or self.turn_stt_final_mono
        if t0 is None:
            return
        now = time.perf_counter()
        parts = [f"gen_start→now={now - t0:.3f}s"]
        if self.turn_llm_first_token_mono is not None:
            parts.append(f"gen_start→llm_1st={self.turn_llm_first_token_mono - t0:.3f}s")
        if self.turn_first_tts_queued_mono is not None:
            parts.append(f"gen_start→tts_q_1st={self.turn_first_tts_queued_mono - t0:.3f}s")
        if self.turn_stt_final_mono is not None and self.generation_anchor_mono is not None:
            parts.append(
                f"stt_final→gen_start={(self.generation_anchor_mono - self.turn_stt_final_mono):.3f}s"
            )
        suf = f" session={session_hint}" if session_hint else ""
        prev = (user_preview[:48] + "…") if len(user_preview) > 48 else user_preview
        log.info("[VoiceMetrics] turn_latency %s user=%r%s", " ".join(parts), prev, suf)
