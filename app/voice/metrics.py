"""
Latency metrics tracking for VoiceOrchestrator V2.

Captures 6 critical latency markers:
1. User speech detected (barge-in controller)
2. STT interim received (Deepgram)
3. LLM speculation started (orchestrator)
4. LLM first token received
5. TTS chunk 1 synthesis complete
6. First audio frame sent to Twilio

Enables accurate latency debugging and performance profiling.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class LatencyMarkers:
    """Captures latency at critical points in voice pipeline."""

    # Timestamps (milliseconds since call start)
    user_speech_detected_ms: int = 0  # Barge-in controller detects first audio
    stt_interim_received_ms: int = 0  # Deepgram sends interim result
    llm_speculation_started_ms: int = 0  # LLM task created
    llm_first_token_ms: int = 0  # First token received from LLM
    tts_chunk_1_ready_ms: int = 0  # First TTS chunk synthesis complete
    first_audio_frame_sent_ms: int = 0  # First audio frame sent to Twilio

    # Computed properties
    @property
    def user_to_interim_ms(self) -> int:
        """Latency: User speech → STT interim result"""
        if self.user_speech_detected_ms == 0 or self.stt_interim_received_ms == 0:
            return 0
        return self.stt_interim_received_ms - self.user_speech_detected_ms

    @property
    def interim_to_llm_first_token_ms(self) -> int:
        """Latency: STT interim → First LLM token"""
        if self.stt_interim_received_ms == 0 or self.llm_first_token_ms == 0:
            return 0
        return self.llm_first_token_ms - self.stt_interim_received_ms

    @property
    def llm_first_token_to_tts_chunk_ms(self) -> int:
        """Latency: LLM first token → TTS chunk synthesis complete"""
        if self.llm_first_token_ms == 0 or self.tts_chunk_1_ready_ms == 0:
            return 0
        return self.tts_chunk_1_ready_ms - self.llm_first_token_ms

    @property
    def tts_chunk_to_audio_frame_ms(self) -> int:
        """Latency: TTS chunk ready → First audio frame sent"""
        if self.tts_chunk_1_ready_ms == 0 or self.first_audio_frame_sent_ms == 0:
            return 0
        return self.first_audio_frame_sent_ms - self.tts_chunk_1_ready_ms

    @property
    def total_first_response_ms(self) -> int:
        """Total latency: User speech → First agent audio heard"""
        if self.user_speech_detected_ms == 0 or self.first_audio_frame_sent_ms == 0:
            return 0
        return self.first_audio_frame_sent_ms - self.user_speech_detected_ms

    def to_dict(self) -> Dict[str, int]:
        """Export all latencies as dictionary"""
        return {
            'user_to_interim_ms': self.user_to_interim_ms,
            'interim_to_llm_first_token_ms': self.interim_to_llm_first_token_ms,
            'llm_first_token_to_tts_chunk_ms': self.llm_first_token_to_tts_chunk_ms,
            'tts_chunk_to_audio_frame_ms': self.tts_chunk_to_audio_frame_ms,
            'total_first_response_ms': self.total_first_response_ms,
        }


class MetricsCollector:
    """
    Collects and logs latency metrics for a single call.
    
    ✅ THE metric that matters: total_first_response_ms
    
    Target: <300ms (ideally 130-200ms)
    """

    def __init__(self, call_id: str):
        self.call_id = call_id
        self.start_time_ms = time.time() * 1000  # Convert to ms
        self.markers = LatencyMarkers()

    def _elapsed_ms(self) -> int:
        """Milliseconds elapsed since call start"""
        return int((time.time() * 1000) - self.start_time_ms)

    def mark_user_speech_detected(self) -> None:
        """Called by barge-in controller when user starts speaking"""
        if self.markers.user_speech_detected_ms == 0:
            self.markers.user_speech_detected_ms = self._elapsed_ms()
            logger.debug(
                f"[{self.call_id}] User speech detected @ {self.markers.user_speech_detected_ms}ms"
            )

    def mark_stt_interim_received(self) -> None:
        """Called by STTStreamManager when interim result arrives"""
        if self.markers.stt_interim_received_ms == 0:
            self.markers.stt_interim_received_ms = self._elapsed_ms()
            elapsed = self.markers.user_to_interim_ms
            logger.debug(
                f"[{self.call_id}] STT interim received @ {self.markers.stt_interim_received_ms}ms "
                f"(+{elapsed}ms from user speech)"
            )

    def mark_llm_speculation_started(self) -> None:
        """Called by VoiceOrchestrator when LLM task created"""
        if self.markers.llm_speculation_started_ms == 0:
            self.markers.llm_speculation_started_ms = self._elapsed_ms()
            logger.debug(
                f"[{self.call_id}] LLM speculation started @ {self.markers.llm_speculation_started_ms}ms"
            )

    def mark_llm_first_token(self) -> None:
        """Called by LLMStreamManager on first LLM token"""
        if self.markers.llm_first_token_ms == 0:
            self.markers.llm_first_token_ms = self._elapsed_ms()
            elapsed = self.markers.interim_to_llm_first_token_ms
            logger.debug(
                f"[{self.call_id}] LLM first token @ {self.markers.llm_first_token_ms}ms "
                f"(+{elapsed}ms from interim)"
            )

    def mark_tts_chunk_1_ready(self) -> None:
        """Called by TTSStreamManager when first chunk synthesis complete"""
        if self.markers.tts_chunk_1_ready_ms == 0:
            self.markers.tts_chunk_1_ready_ms = self._elapsed_ms()
            elapsed = self.markers.llm_first_token_to_tts_chunk_ms
            logger.debug(
                f"[{self.call_id}] TTS chunk 1 ready @ {self.markers.tts_chunk_1_ready_ms}ms "
                f"(+{elapsed}ms from LLM token)"
            )

    def mark_first_audio_frame_sent(self) -> None:
        """Called by VoiceOrchestrator when first frame sent to Twilio"""
        if self.markers.first_audio_frame_sent_ms == 0:
            self.markers.first_audio_frame_sent_ms = self._elapsed_ms()
            elapsed = self.markers.tts_chunk_to_audio_frame_ms
            total = self.markers.total_first_response_ms
            logger.debug(
                f"[{self.call_id}] First audio frame sent @ {self.markers.first_audio_frame_sent_ms}ms "
                f"(+{elapsed}ms from TTS chunk)"
            )
            logger.info(
                f"[{self.call_id}] 🎯 TOTAL FIRST RESPONSE: {total}ms "
                f"(target: <300ms, ideal: 130-200ms) "
                f"{'✅ PASS' if total < 300 else '❌ FAIL'}"
            )

    def get_latency_summary(self) -> Dict:
        """Export complete latency summary for logging/telemetry"""
        return {
            'call_id': self.call_id,
            'markers': {
                'user_speech_detected_ms': self.markers.user_speech_detected_ms,
                'stt_interim_received_ms': self.markers.stt_interim_received_ms,
                'llm_speculation_started_ms': self.markers.llm_speculation_started_ms,
                'llm_first_token_ms': self.markers.llm_first_token_ms,
                'tts_chunk_1_ready_ms': self.markers.tts_chunk_1_ready_ms,
                'first_audio_frame_sent_ms': self.markers.first_audio_frame_sent_ms,
            },
            'latencies': self.markers.to_dict(),
        }

    def log_summary(self, level: str = 'info') -> None:
        """Log complete latency summary"""
        summary = self.get_latency_summary()
        log_fn = getattr(logger, level)
        log_fn(f"[{self.call_id}] Latency summary: {summary}")
