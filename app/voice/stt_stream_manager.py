"""
STTStreamManager: Async Deepgram streaming adapter for VoiceOrchestrator V2.

Replaces SttPipeline (V1 queue-based) with direct async event emission.

Key improvements over V1:
- Interim events fire to orchestrator (enabling early LLM speculation)
- Adaptive endpointing (normal / extended / aggressive)
- 6-second deduplication window (prevents Twilio re-endpoint duplicates)
- Long-lived reusable session (no reconnect per utterance)
"""

import asyncio
import hashlib
import logging
import time
from typing import TYPE_CHECKING, Optional

from app.core.config import settings
from app.services.deepgram_stt_service import deepgram_stt_service

if TYPE_CHECKING:
    from app.voice.orchestrator import VoiceOrchestrator

logger = logging.getLogger(__name__)


class EndpointingMode:
    """Adaptive endpointing modes for Deepgram silence detection."""
    NORMAL = "normal"          # 300ms — default conversational
    EXTENDED = "extended"      # 600ms — email/number spelling
    AGGRESSIVE = "aggressive"  # 100ms — high-confidence barge-in path


class STTStreamManager:
    """
    Manages Deepgram WebSocket session and routes STT events to VoiceOrchestrator.

    Event flow:
      feed_audio(pcm) → Deepgram → interim/final callbacks → orchestrator.on_stt_*()

    Thread safety: All methods are async. Audio feeding is non-blocking (push-only).
    """

    # Deduplication window: reject identical finals within this period
    DEDUP_WINDOW_SEC: float = 6.0

    def __init__(self, call_id: str, orchestrator: "VoiceOrchestrator") -> None:
        self.call_id = call_id
        self.orchestrator = orchestrator

        self._language_code: Optional[str] = None
        self._endpointing_mode: str = EndpointingMode.NORMAL
        self._endpointing_ms: int = settings.DEEPGRAM_STT_ENDPOINTING_MS

        self._stt_session = None
        self._reader_task: Optional[asyncio.Task] = None

        # Deduplication: hash → epoch timestamp
        self._dedup_window: dict[str, float] = {}

        # Word count from latest interim (for early LLM trigger gating)
        self._current_word_count: int = 0
        self._current_interim: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def configure(self, language_code: Optional[str] = None) -> None:
        """Set language and other session parameters before start."""
        self._language_code = language_code

    async def start(self) -> None:
        """Initialize Deepgram session. Call once at orchestrator startup."""
        await self._ensure_session()
        logger.info(f"[{self.call_id}] STTStreamManager started (mode={self._endpointing_mode})")

    async def feed_audio(self, mulaw_chunk: bytes) -> None:
        """
        Feed raw MULAW audio chunk from Twilio into Deepgram.

        Non-blocking: pushes to session's internal queue.
        Lazily starts session on first call if not already started.
        """
        if not mulaw_chunk:
            return
        await self._ensure_session()
        if self._stt_session:
            self._stt_session.push_audio(mulaw_chunk)

    def set_endpointing_mode(self, mode: str) -> None:
        """
        Adjust silence detection threshold at runtime.

        Args:
            mode: EndpointingMode.NORMAL / EXTENDED / AGGRESSIVE
        """
        self._endpointing_mode = mode
        if mode == EndpointingMode.EXTENDED:
            self._endpointing_ms = settings.DEEPGRAM_STT_ENDPOINTING_MS_EXTENDED
        elif mode == EndpointingMode.AGGRESSIVE:
            self._endpointing_ms = 100
        else:
            self._endpointing_ms = settings.DEEPGRAM_STT_ENDPOINTING_MS

        logger.info(
            f"[{self.call_id}] STT endpointing → {mode} ({self._endpointing_ms}ms)"
        )

    async def recreate_with_endpointing(self, endpointing_ms: int) -> None:
        """
        Re-open Deepgram session with new endpointing value.

        Used when agent asks for email/number so spelling pauses don't split finals.
        """
        await self.stop()
        self._endpointing_ms = endpointing_ms
        self._stt_session = None
        self._reader_task = None
        await self._ensure_session()
        logger.info(
            f"[{self.call_id}] STT session recreated with endpointing={endpointing_ms}ms"
        )

    async def stop(self) -> None:
        """
        Gracefully close Deepgram session.

        Signals finish → waits for reader to consume final {done:True} → cancels.
        """
        if self._stt_session:
            try:
                self._stt_session.finish()
            except Exception:
                pass

        if self._reader_task and not self._reader_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._reader_task), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"[{self.call_id}] STT reader did not finish in 5s — cancelling")
                self._reader_task.cancel()
                try:
                    await self._reader_task
                except (asyncio.CancelledError, Exception):
                    pass
            except asyncio.CancelledError:
                pass

        self._stt_session = None
        self._reader_task = None
        logger.info(f"[{self.call_id}] STTStreamManager stopped")

    # ------------------------------------------------------------------
    # Internal session management
    # ------------------------------------------------------------------

    async def _ensure_session(self) -> None:
        """Lazily start Deepgram session on first audio feed."""
        if self._stt_session is not None:
            return

        self._stt_session = deepgram_stt_service.create_streaming_session(
            language_code=self._language_code,
            encoding="MULAW",
            sample_rate=8000,
            interim_results=True,
            single_utterance=False,
            endpointing_ms=self._endpointing_ms,
        )

        async def _consume_start():
            """Start the underlying blocking stream in executor."""
            try:
                await self._stt_session.start()
            except Exception as e:
                logger.error(f"[{self.call_id}] STT session start error: {e}", exc_info=True)

        async def _reader_loop():
            """Consume Deepgram results and route to orchestrator."""
            while True:
                sess = self._stt_session
                if sess is None:
                    break
                try:
                    result = await sess.get_result()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"[{self.call_id}] STT reader error: {e}", exc_info=True)
                    if self._stt_session is None:
                        break
                    continue

                if not result:
                    continue
                if result.get("done"):
                    break
                if result.get("error"):
                    logger.warning(
                        f"[{self.call_id}] STT error payload: {result.get('error')}"
                    )
                    continue

                transcript = (result.get("transcript") or "").strip()
                if not transcript:
                    continue

                is_final = bool(result.get("is_final"))
                confidence = float(result.get("confidence") or 0.0)

                try:
                    if is_final:
                        await self._on_final(transcript, confidence)
                    else:
                        await self._on_interim(transcript, confidence)
                except Exception as cb_err:
                    logger.error(
                        f"[{self.call_id}] STT callback error: {cb_err}", exc_info=True
                    )

        self._reader_task = asyncio.create_task(_reader_loop())
        asyncio.create_task(_consume_start())

    # ------------------------------------------------------------------
    # Event handlers → orchestrator dispatch
    # ------------------------------------------------------------------

    async def _on_interim(self, text: str, confidence: float) -> None:
        """
        Interim transcript received from Deepgram.

        Updates internal word count cache and fires to orchestrator.
        Orchestrator will start LLM speculation if word_count >= 3.
        """
        words = text.split()
        self._current_word_count = len(words)
        self._current_interim = text

        logger.debug(
            f"[{self.call_id}] STT interim: '{text}' "
            f"({self._current_word_count}w @ {confidence:.2f})"
        )

        await self.orchestrator.on_stt_interim(
            text=text,
            confidence=confidence,
            word_count=self._current_word_count,
        )

    async def _on_final(self, text: str, confidence: float) -> None:
        """
        Final transcript received from Deepgram.

        Applies deduplication (6s window) and minimum confidence gate
        before routing to orchestrator.
        """
        # --- Minimum confidence gate ---
        min_confidence = settings.VOICE_STT_MIN_FINAL_CONFIDENCE
        if confidence < min_confidence:
            # Soft fallback: multi-word alpha content may still be valid
            if settings.VOICE_STT_ENABLE_SOFT_FINAL_FALLBACK:
                words = text.split()
                alpha_words = [w for w in words if any(c.isalpha() for c in w)]
                if (
                    confidence >= settings.VOICE_STT_SOFT_MIN_FINAL_CONFIDENCE
                    and len(alpha_words) >= settings.VOICE_STT_SOFT_MIN_WORDS
                ):
                    logger.debug(
                        f"[{self.call_id}] STT soft-fallback accepted: "
                        f"'{text}' @ {confidence:.2f}"
                    )
                else:
                    logger.debug(
                        f"[{self.call_id}] STT final rejected (low confidence): "
                        f"'{text}' @ {confidence:.2f} < {min_confidence}"
                    )
                    return
            else:
                return

        # --- Deduplication ---
        if self._is_duplicate(text):
            logger.debug(f"[{self.call_id}] STT final deduplicated: '{text}'")
            return

        logger.info(
            f"[{self.call_id}] STT final: '{text}' @ {confidence:.2f}"
        )
        await self.orchestrator.on_stt_final(text=text, confidence=confidence)

    def _is_duplicate(self, text: str) -> bool:
        """
        Check if this final transcript was seen within the dedup window.

        Cleans up expired entries on every call (no background task needed).
        """
        now = time.monotonic()

        # Expire old entries
        expired = [k for k, ts in self._dedup_window.items() if now - ts > self.DEDUP_WINDOW_SEC]
        for k in expired:
            del self._dedup_window[k]

        # Hash the transcript for memory-efficient storage
        key = hashlib.md5(text.lower().strip().encode()).hexdigest()

        if key in self._dedup_window:
            return True

        self._dedup_window[key] = now
        return False
