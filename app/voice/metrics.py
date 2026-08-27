"""
Lightweight latency markers for voice (STT → LLM → TTS). Observability only.

All timestamps use time.perf_counter() for monotonic deltas. No I/O in hot paths.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class VoiceTurnMetrics:
    """Per-call metrics with one active user turn at a time."""

    turn_id: str | int | None = None
    transport: str = "telephony"
    agent_id: str | None = None
    provider: str | None = None
    call_sid: str | None = None

    call_pickup_mono: float | None = None
    acoustic_speech_end_mono: float | None = None
    stt_final_packet_received_mono: float | None = None
    turn_stt_final_mono: float | None = None
    generation_start_mono: float | None = None
    generation_anchor_mono: float | None = None
    rag_start_mono: float | None = None
    rag_end_mono: float | None = None
    prompt_start_mono: float | None = None
    prompt_ready_mono: float | None = None
    llm_request_mono: float | None = None
    turn_llm_first_token_mono: float | None = None
    turn_first_tts_queued_mono: float | None = None
    tts_first_audio_mono: float | None = None
    first_playback_mono: float | None = None
    turn_complete_mono: float | None = None

    def __init__(
        self,
        call_sid: str | None = None,
        turn_id: str | int | None = None,
        transport: str = "telephony",
        agent_id: str | None = None,
        provider: str | None = None,
    ) -> None:
        self.call_sid = call_sid
        self.turn_id = turn_id
        self.transport = transport
        self.agent_id = agent_id
        self.provider = provider
        self.call_pickup_mono = None
        self.acoustic_speech_end_mono = None
        self.stt_final_packet_received_mono = None
        self.turn_stt_final_mono = None
        self.generation_start_mono = None
        self.generation_anchor_mono = None
        self.rag_start_mono = None
        self.rag_end_mono = None
        self.prompt_start_mono = None
        self.prompt_ready_mono = None
        self.llm_request_mono = None
        self.turn_llm_first_token_mono = None
        self.turn_first_tts_queued_mono = None
        self.tts_first_audio_mono = None
        self.first_playback_mono = None
        self.turn_complete_mono = None
        # Internal per-turn sequence counter backing self.turn_id. Not itself
        # a dataclass field — start_generation() bumps it and republishes the
        # new value into self.turn_id so telemetry/log consumers and any
        # caller wanting to validate turn ownership (see mark_rag_start /
        # mark_rag_end / mark_llm_request below) see a real, monotonically
        # increasing per-call turn counter instead of the permanently-None
        # default the field previously carried.
        self._turn_seq: int = 0

    def mark_call_pickup(self) -> None:
        if self.call_pickup_mono is None:
            self.call_pickup_mono = time.perf_counter()

    def record_acoustic_speech_end(self, timestamp_mono: float) -> None:
        """Record the acoustic speech end time (from STT word-level metadata / audio stream offset)."""
        self.acoustic_speech_end_mono = timestamp_mono

    def begin_turn_at_stt_final(self, acoustic_speech_end_mono: float | None = None) -> None:
        """Record accepted final STT packet receive time and optional acoustic speech end timestamp."""
        now = time.perf_counter()
        self.stt_final_packet_received_mono = now
        self.turn_stt_final_mono = now
        if acoustic_speech_end_mono is not None:
            self.acoustic_speech_end_mono = acoustic_speech_end_mono

    def start_generation(self) -> int:
        """Call once at the start of each turn's generation, BEFORE any
        mark_* calls for that turn (rag/llm/tts marks included) — this is
        the single per-turn reset point. Returns the new turn_id so callers
        that fire detached background tasks (e.g. RAG prefetch, which starts
        speculatively on an interim transcript and can outlive the turn that
        eventually consumes it, or gets cancelled by a barge-in) can snapshot
        "the turn I was started for" and pass it back into mark_rag_start /
        mark_rag_end / mark_llm_request so a write arriving after the turn
        has already moved on is rejected instead of silently corrupting the
        new turn's fields.

        Resets every field that is turn-scoped. rag_start_mono / rag_end_mono
        / llm_request_mono are deliberately included here — historically they
        were NOT reset, which meant their write-once-per-call idempotent
        guards (`if self.x is None`) only ever fired on the call's first
        turn and then stayed frozen for the rest of the call, corrupting
        every later turn's rag_latency_ms / llm_ttft_ms.
        """
        now = time.perf_counter()
        self._turn_seq += 1
        self.turn_id = self._turn_seq
        self.generation_start_mono = now
        self.generation_anchor_mono = now
        self.prompt_start_mono = now
        self.rag_start_mono = None
        self.rag_end_mono = None
        self.llm_request_mono = None
        self.turn_llm_first_token_mono = None
        self.turn_first_tts_queued_mono = None
        self.tts_first_audio_mono = None
        self.first_playback_mono = None
        self.turn_complete_mono = None
        return self.turn_id

    def mark_rag_start(self, expected_turn_id: int | None = None) -> None:
        """expected_turn_id: reserved for a detached background task to pass
        the turn_id it snapshotted when scheduled (e.g. speculative RAG
        prefetch fired from an interim transcript, before start_generation()
        for the eventual real turn may even have run) — if the current
        turn_id has since moved on (barge-in / new turn started), the write
        is dropped rather than corrupting the new turn's fields. NOT YET
        WIRED to any production caller as of this writing (the RAG-prefetch
        background task does not call mark_rag_start()/mark_llm_request() at
        all today, so this parameter is always None in practice) — kept as
        forward-looking infrastructure for whichever caller ends up needing
        it. Do not assume this guard is currently protecting anything."""
        if expected_turn_id is not None and expected_turn_id != self.turn_id:
            return
        if self.rag_start_mono is None:
            self.rag_start_mono = time.perf_counter()

    def mark_rag_end(self, expected_turn_id: int | None = None) -> None:
        if expected_turn_id is not None and expected_turn_id != self.turn_id:
            return
        self.rag_end_mono = time.perf_counter()

    def mark_prompt_ready(self) -> None:
        self.prompt_ready_mono = time.perf_counter()

    def mark_llm_request(self, expected_turn_id: int | None = None) -> None:
        if expected_turn_id is not None and expected_turn_id != self.turn_id:
            return
        if self.llm_request_mono is None:
            self.llm_request_mono = time.perf_counter()

    def mark_llm_first_token(self) -> None:
        if self.turn_llm_first_token_mono is None:
            self.turn_llm_first_token_mono = time.perf_counter()

    def mark_first_tts_queued(self) -> None:
        if self.turn_first_tts_queued_mono is None:
            self.turn_first_tts_queued_mono = time.perf_counter()

    def mark_tts_first_audio(self) -> None:
        if self.tts_first_audio_mono is None:
            self.tts_first_audio_mono = time.perf_counter()

    def mark_first_playback(self) -> None:
        if self.first_playback_mono is None:
            self.first_playback_mono = time.perf_counter()

    def mark_turn_complete(self) -> None:
        self.turn_complete_mono = time.perf_counter()

    def mark_live_first_audio(self) -> None:
        """
        Gemini Live (speech-to-speech native-audio) marker: collapses what
        would otherwise be two separate markers into one first audio byte event.
        """
        now = time.perf_counter()
        if self.turn_llm_first_token_mono is None:
            self.turn_llm_first_token_mono = now
        if self.turn_first_tts_queued_mono is None:
            self.turn_first_tts_queued_mono = now
        if self.tts_first_audio_mono is None:
            self.tts_first_audio_mono = now
        if self.first_playback_mono is None:
            self.first_playback_mono = now

    def calculate_latencies(self) -> dict[str, float | None]:
        """Compute stage deltas in milliseconds using canonical monotonic anchors."""
        stt_final_mono = self.stt_final_packet_received_mono or self.turn_stt_final_mono
        gen_start_mono = self.generation_start_mono or self.generation_anchor_mono

        # 1. Acoustic endpointing: acoustic speech end -> STT final packet received
        acoustic_endpointing_ms = None
        if self.acoustic_speech_end_mono is not None and stt_final_mono is not None:
            acoustic_endpointing_ms = max(0.0, (stt_final_mono - self.acoustic_speech_end_mono) * 1000)

        # 2. STT dispatch latency: STT final packet received -> generation start
        stt_dispatch_ms = None
        if stt_final_mono is not None and gen_start_mono is not None:
            stt_dispatch_ms = max(0.0, (gen_start_mono - stt_final_mono) * 1000)

        # 3. RAG retrieval latency: rag_start -> rag_end
        rag_latency_ms = None
        if self.rag_start_mono is not None and self.rag_end_mono is not None:
            rag_latency_ms = max(0.0, (self.rag_end_mono - self.rag_start_mono) * 1000)

        # 4. Prompt assembly latency: prompt_start -> prompt_ready
        prompt_assembly_ms = None
        if self.prompt_start_mono is not None and self.prompt_ready_mono is not None:
            prompt_assembly_ms = max(0.0, (self.prompt_ready_mono - self.prompt_start_mono) * 1000)

        # 5. LLM Time to First Token (TTFT): llm_request -> first token
        llm_ttft_ms = None
        llm_base = self.llm_request_mono or self.prompt_ready_mono or gen_start_mono
        if llm_base is not None and self.turn_llm_first_token_mono is not None:
            llm_ttft_ms = max(0.0, (self.turn_llm_first_token_mono - llm_base) * 1000)

        # 6. TTS Time to First Audio (TTFA): chunk queued -> first audio byte
        tts_ttfa_ms = None
        if self.turn_first_tts_queued_mono is not None and self.tts_first_audio_mono is not None:
            tts_ttfa_ms = max(0.0, (self.tts_first_audio_mono - self.turn_first_tts_queued_mono) * 1000)

        # 7. Playback gate wait: first audio ready -> playback begins
        playback_gate_wait_ms = None
        if self.tts_first_audio_mono is not None and self.first_playback_mono is not None:
            playback_gate_wait_ms = max(0.0, (self.first_playback_mono - self.tts_first_audio_mono) * 1000)

        # 8. End-to-end turnaround: STT final -> first audio byte
        stt_final_to_first_audio_ms = None
        audio_anchor = self.tts_first_audio_mono or self.first_playback_mono
        if stt_final_mono is not None and audio_anchor is not None:
            stt_final_to_first_audio_ms = max(0.0, (audio_anchor - stt_final_mono) * 1000)

        # 8b. STT final -> first playback
        stt_final_to_first_playback_ms = None
        if stt_final_mono is not None and self.first_playback_mono is not None:
            stt_final_to_first_playback_ms = max(0.0, (self.first_playback_mono - stt_final_mono) * 1000)

        # 9. True acoustic turnaround: acoustic speech end -> first playback
        acoustic_end_to_first_playback_ms = None
        if self.acoustic_speech_end_mono is not None and self.first_playback_mono is not None:
            acoustic_end_to_first_playback_ms = max(0.0, (self.first_playback_mono - self.acoustic_speech_end_mono) * 1000)

        # 10. Total turn latency
        total_turn_latency_ms = None
        t_end = self.first_playback_mono or self.tts_first_audio_mono or self.turn_complete_mono
        t_start = stt_final_mono or gen_start_mono
        if t_start is not None and t_end is not None:
            total_turn_latency_ms = max(0.0, (t_end - t_start) * 1000)

        return {
            "acoustic_endpointing_ms": round(acoustic_endpointing_ms, 1) if acoustic_endpointing_ms is not None else None,
            "stt_dispatch_ms": round(stt_dispatch_ms, 1) if stt_dispatch_ms is not None else None,
            "stt_endpoint_latency_ms": round(acoustic_endpointing_ms or stt_dispatch_ms, 1) if (acoustic_endpointing_ms is not None or stt_dispatch_ms is not None) else None,
            "rag_latency_ms": round(rag_latency_ms, 1) if rag_latency_ms is not None else None,
            "prompt_assembly_latency_ms": round(prompt_assembly_ms, 1) if prompt_assembly_ms is not None else None,
            "llm_ttft_ms": round(llm_ttft_ms, 1) if llm_ttft_ms is not None else None,
            "tts_ttfa_ms": round(tts_ttfa_ms, 1) if tts_ttfa_ms is not None else None,
            "playback_gate_wait_ms": round(playback_gate_wait_ms, 1) if playback_gate_wait_ms is not None else None,
            "stt_final_to_first_audio_ms": round(stt_final_to_first_audio_ms, 1) if stt_final_to_first_audio_ms is not None else None,
            "stt_final_to_first_playback_ms": round(stt_final_to_first_playback_ms, 1) if stt_final_to_first_playback_ms is not None else None,
            "acoustic_end_to_first_playback_ms": round(acoustic_end_to_first_playback_ms, 1) if acoustic_end_to_first_playback_ms is not None else None,
            "total_turn_latency_ms": round(total_turn_latency_ms, 1) if total_turn_latency_ms is not None else None,
        }

    def build_telemetry_payload(self, user_preview: str = "") -> dict[str, Any]:
        """Build structured JSON telemetry dictionary without sensitive text content."""
        latencies = self.calculate_latencies()
        return {
            "telemetry_type": "voice_turn",
            "call_sid": self.call_sid,
            "turn_id": self.turn_id,
            "transport": self.transport,
            "agent_id": self.agent_id,
            "provider": self.provider,
            "user_preview": user_preview,
            "timestamps_mono": {
                "acoustic_speech_end": self.acoustic_speech_end_mono,
                "stt_final_packet": self.stt_final_packet_received_mono or self.turn_stt_final_mono,
                "generation_start": self.generation_start_mono or self.generation_anchor_mono,
                "rag_start": self.rag_start_mono,
                "rag_end": self.rag_end_mono,
                "prompt_start": self.prompt_start_mono,
                "prompt_ready": self.prompt_ready_mono,
                "llm_request": self.llm_request_mono,
                "llm_first_token": self.turn_llm_first_token_mono,
                "tts_queue": self.turn_first_tts_queued_mono,
                "tts_first_audio": self.tts_first_audio_mono,
                "first_playback": self.first_playback_mono,
                "turn_complete": self.turn_complete_mono,
            },
            "latencies": latencies,
            "latencies_ms": latencies,
        }

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

        # Also emit structured telemetry payload for metrics pipeline
        try:
            payload = self.build_telemetry_payload()
            log.info("[VoiceTelemetry] %s", payload)
        except Exception:
            pass

    def get_slo_breaches(
        self,
        *,
        stt_final_to_gen_start_s: float,
        gen_start_to_llm_first_token_s: float,
        gen_start_to_first_tts_q_s: float,
        gen_start_to_now_warn_s: float,
    ) -> list[str]:
        """
        Return human-readable SLO breach strings for the current generation.
        Missing markers are ignored (best-effort observability).
        """
        out: list[str] = []
        t0 = self.generation_anchor_mono
        if t0 is None:
            return out

        now = time.perf_counter()
        elapsed = now - t0
        if elapsed > gen_start_to_now_warn_s:
            out.append(
                f"gen_start→now {elapsed:.3f}s > {gen_start_to_now_warn_s:.3f}s"
            )

        if (
            self.turn_stt_final_mono is not None
            and self.generation_anchor_mono is not None
        ):
            stt_to_gen = self.generation_anchor_mono - self.turn_stt_final_mono
            if stt_to_gen > stt_final_to_gen_start_s:
                out.append(
                    f"stt_final→gen_start {stt_to_gen:.3f}s > {stt_final_to_gen_start_s:.3f}s"
                )

        if self.turn_llm_first_token_mono is not None:
            llm = self.turn_llm_first_token_mono - t0
            if llm > gen_start_to_llm_first_token_s:
                out.append(
                    f"gen_start→llm_1st {llm:.3f}s > {gen_start_to_llm_first_token_s:.3f}s"
                )

        if self.turn_first_tts_queued_mono is not None:
            tts = self.turn_first_tts_queued_mono - t0
            if tts > gen_start_to_first_tts_q_s:
                out.append(
                    f"gen_start→tts_q_1st {tts:.3f}s > {gen_start_to_first_tts_q_s:.3f}s"
                )
        return out
