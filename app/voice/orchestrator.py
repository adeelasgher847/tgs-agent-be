"""
VoiceOrchestrator V2: Central event-driven brain for voice calls.

Replaces the 3500-line bidirectional_stream.py handler with a flat,
minimal-latency, pure async architecture.

Architecture:
  Twilio WebSocket
    ↓
  VoiceOrchestrator.process_twilio_frame()
    ↓ feeds audio
  STTStreamManager → interim/final events → back to orchestrator
    ↓ on_stt_interim (3+ words)
  asyncio.create_task(LLMStreamManager.stream_speculative())  ← NON-BLOCKING
    ↓ on_llm_chunk (2-12 word batch)
  asyncio.create_task(TTSStreamManager.enqueue_chunk())       ← NON-BLOCKING
    ↓ on_tts_frame_ready
  send_twilio_frame()

Barge-in path (< 100ms):
  STTStreamManager.interim → BargeInController.check_trigger()
    → on_barge_in() → cancellation_token.cancel_all()
    → All LLM/TTS tasks receive CancelledError → stop immediately

Key design principles:
  - NO asyncio.Queue in core voice path
  - All events are direct method calls (no event bus layer)
  - CancellationToken is per-turn (reset after barge-in)
  - State is single source of truth (ConversationStateManager)
"""

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from app.voice.cancellation import CancellationToken
from app.voice.conversation_state_manager import ConversationState, ConversationStateManager
from app.voice.stt_stream_manager import STTStreamManager, EndpointingMode
from app.voice.llm_stream_manager import LLMStreamManager
from app.voice.tts_stream_manager import TTSStreamManager
from app.voice.barge_in_controller import BargeInController

logger = logging.getLogger(__name__)


class VoiceOrchestrator:
    """
    Central event-driven orchestrator for a single voice call.

    Lifecycle:
      __init__() → start_call() → [process_twilio_frame() loop] → shutdown()

    Thread safety:
      All methods are async — must be awaited from the WebSocket handler coroutine.
      No threads (except executor for sync TTS providers).
    """

    def __init__(
        self,
        call_id: str,
        agent_id: str,
        agent_config: Dict[str, Any],
        send_twilio_frame_callback: Callable[[bytes], Any],
    ) -> None:
        """
        Initialize orchestrator and all sub-managers.

        Args:
            call_id: Unique call session ID (UUID string).
            agent_id: Agent UUID.
            agent_config: Dict with agent, tts_provider, language, etc.
            send_twilio_frame_callback: Coroutine or callable to send MULAW bytes to Twilio.
        """
        self.call_id = call_id
        self.agent_id = agent_id
        self.agent_config = agent_config
        self._send_twilio_frame = send_twilio_frame_callback

        # --- Core state ---
        self.state_mgr = ConversationStateManager(call_id, agent_config)

        # --- Sub-managers (initialized in __init__, not lazily) ---
        self.stt_mgr = STTStreamManager(call_id, self)
        self.llm_mgr = LLMStreamManager(call_id, agent_config, self)
        self.tts_mgr = TTSStreamManager(call_id, agent_config, self)
        self.barge_in_ctrl = BargeInController(call_id, self)

        # --- Per-turn cancellation token ---
        # Replaced on every barge-in so new tasks get a clean token
        self.cancellation_token: CancellationToken = CancellationToken(call_id)

        # --- Turn tracking ---
        self._speculation_started: bool = False
        self._llm_started_at_ms: int = 0
        self._call_active: bool = False

        # --- Early LLM trigger threshold (from settings) ---
        from app.core.config import settings
        self._early_llm_min_words: int = getattr(settings, "VOICE_MIN_INTERIM_WORDS", 3)
        self._early_llm_min_confidence: float = getattr(
            settings, "VOICE_MIN_INTERIM_CONFIDENCE", 0.18
        )

        # --- Configure TTS provider slug in LLM manager for SSML ---
        agent = agent_config.get("agent")
        if agent:
            tts_provider = getattr(agent, "tts_provider", None)
            if tts_provider:
                slug = (getattr(tts_provider, "slug", "") or "").lower()
                self.llm_mgr.configure_tts_provider(slug)

            # Configure STT language
            lang = (getattr(agent, "language", None) or "en")
            self.stt_mgr.configure(language_code=lang)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start_call(self) -> None:
        """
        Initialize all managers and start the STT session.

        Call once after WebSocket connection is established.
        """
        await self.state_mgr.transition_state(ConversationState.WAITING_FOR_INPUT)
        await self.stt_mgr.start()
        self._call_active = True
        logger.info(f"[{self.call_id}] VoiceOrchestrator V2 started")

    async def shutdown(self) -> None:
        """
        Gracefully shut down all managers.

        Call when WebSocket disconnects or call ends.
        """
        if not self._call_active:
            return

        self._call_active = False
        await self.cancellation_token.cancel_all(timeout_ms=200)

        await asyncio.gather(
            self.stt_mgr.stop(),
            self.llm_mgr.stop(),
            self.tts_mgr.stop(),
            return_exceptions=True,
        )
        await self.state_mgr.transition_state(ConversationState.CALL_ENDED)
        logger.info(f"[{self.call_id}] VoiceOrchestrator V2 shut down")

    # ------------------------------------------------------------------
    # Main ingestion entry point
    # ------------------------------------------------------------------

    async def process_twilio_frame(self, mulaw_frame: bytes) -> None:
        """
        Entry point: receive a 20ms MULAW frame from Twilio WebSocket.

        Feeds audio to STT. STT results come back via on_stt_interim/on_stt_final.
        This method must be called for every inbound Twilio media frame.

        Non-blocking: audio push is synchronous, STT processing is async.
        """
        if not self._call_active:
            return
        await self.stt_mgr.feed_audio(mulaw_frame)

    # ------------------------------------------------------------------
    # STT event handlers (called by STTStreamManager)
    # ------------------------------------------------------------------

    async def on_stt_interim(
        self, text: str, confidence: float, word_count: int
    ) -> None:
        """
        Interim transcript received from Deepgram.

        Two responsibilities:
        1. Early LLM speculation: start on 3+ words before final arrives
        2. Barge-in detection: if agent is speaking, check for interrupt

        CRITICAL: asyncio.create_task() here — NOT blocking.
        """
        self.state_mgr.set_interim_text(text)

        # --- Barge-in check (highest priority) ---
        if self.state_mgr.agent_is_speaking:
            await self.barge_in_ctrl.check_trigger(text, confidence)
            return  # Don't start speculation while agent is speaking

        # --- Early LLM speculation ---
        if (
            not self._speculation_started
            and word_count >= self._early_llm_min_words
            and confidence >= self._early_llm_min_confidence
        ):
            await self._start_speculation(text)

    async def on_stt_final(self, text: str, confidence: float) -> None:
        """
        Final transcript received from Deepgram.

        NOT blocking — updates state, handles divergence from speculation.

        Divergence logic:
        - If speculation ran with interim text that differs significantly
          from final → cancel speculation + re-run with final
        - "Significant" = final text does NOT start with interim prefix
        """
        self.state_mgr.set_final_text(text)
        await self.state_mgr.transition_state(ConversationState.PROCESSING)

        # Add user turn to history
        await self.state_mgr.add_to_history(
            role="user",
            text=text,
            confidence=confidence,
        )

        # Check if speculation diverged
        interim = self.state_mgr.interim_text
        diverged = self.llm_mgr.has_speculation and not text.lower().startswith(
            interim.lower()[: max(len(interim) // 2, 4)]
        )

        if diverged:
            logger.info(
                f"[{self.call_id}] STT final diverges from interim — re-running LLM. "
                f"Interim: '{interim[:30]}' | Final: '{text[:30]}'"
            )
            system_prompt = self._build_system_prompt()
            history = self.state_mgr.get_messages_for_llm()
            await self.llm_mgr.finalize_and_rerun(
                final_text=text,
                cancellation_token=self.cancellation_token,
                conversation_history=history,
                system_prompt=system_prompt,
            )
        elif not self._speculation_started:
            # No speculation yet (very short interim or low confidence) → start now
            await self._start_speculation(text)

        # Reset speculation flag for next turn
        self._speculation_started = False

    # ------------------------------------------------------------------
    # LLM event handlers (called by LLMStreamManager)
    # ------------------------------------------------------------------

    async def on_llm_chunk(
        self,
        text: str,
        is_final: bool,
        end_call_after: bool,
        is_quick_ack: bool = False,
    ) -> None:
        """
        LLM produced a voice-ready text chunk.

        Transition to AgentSpeaking on first chunk, then spawn TTS task.
        NON-BLOCKING: TTS synthesis runs as background task.
        """
        if self.cancellation_token.is_cancelled():
            return

        # Mark agent as speaking on first content chunk
        if not self.state_mgr.agent_is_speaking and not is_quick_ack:
            self.state_mgr.agent_speaking_start()
            self.barge_in_ctrl.arm()
            await self.state_mgr.transition_state(ConversationState.AGENT_SPEAKING)

        # Spawn TTS synthesis as background task (non-blocking)
        tts_task = asyncio.create_task(
            self.tts_mgr.enqueue_chunk(
                text=text,
                cancellation_token=self.cancellation_token,
                is_final=is_final,
                is_quick_ack=is_quick_ack,
                end_call_after=end_call_after,
            ),
            name=f"llm_chunk_tts_{self.call_id}",
        )
        await self.cancellation_token.register_task(tts_task)

    async def on_llm_end_call(self) -> None:
        """
        LLM signalled [END_CALL] but with no final text.

        Trigger call end after current TTS completes.
        """
        logger.info(f"[{self.call_id}] LLM requested end call (no final text)")
        await self.on_tts_complete(end_call_after=True)

    # ------------------------------------------------------------------
    # TTS event handlers (called by TTSStreamManager)
    # ------------------------------------------------------------------

    async def on_tts_frame_ready(self, mulaw_frame: bytes) -> None:
        """
        TTS produced a 20ms MULAW frame — forward to Twilio immediately.

        This is the hot path: must not block.
        """
        try:
            await self._send_twilio_frame(mulaw_frame)
        except Exception as e:
            logger.error(f"[{self.call_id}] Failed to send Twilio frame: {e}")

    async def on_tts_complete(self, end_call_after: bool = False) -> None:
        """
        TTS finished streaming all chunks for this agent turn.

        Resets state, disarms barge-in, transitions back to waiting.
        """
        self.state_mgr.agent_speaking_end()
        self.barge_in_ctrl.disarm()

        # Add agent turn to history
        # (text accumulated by LLM manager — we use interim_text as proxy)
        # Full text tracking is handled per-turn via llm_mgr response_accum
        await self.state_mgr.transition_state(ConversationState.WAITING_FOR_INPUT)

        logger.debug(f"[{self.call_id}] TTS complete, waiting for next input")

        if end_call_after:
            logger.info(f"[{self.call_id}] End call requested — shutting down")
            await self.shutdown()

    # ------------------------------------------------------------------
    # Barge-in handler (called by BargeInController)
    # ------------------------------------------------------------------

    async def on_barge_in(self) -> None:
        """
        User interrupted agent speech.

        Cancels all in-flight LLM + TTS tasks instantly.
        Resets state so user can speak.

        Target: < 100ms from detection to audio stop.
        """
        logger.info(f"[{self.call_id}] ⚡ Barge-in detected — cancelling all tasks")

        # Cancel all running tasks (LLM + TTS)
        await self.cancellation_token.cancel_all(timeout_ms=100)

        # Reset agent speaking state
        self.state_mgr.cancel_agent_speaking()
        self.tts_mgr.reset_for_new_turn()
        self.barge_in_ctrl.disarm()

        # Create fresh token for next turn
        self.cancellation_token = CancellationToken(self.call_id)

        # Transition back to waiting
        await self.state_mgr.transition_state(ConversationState.INTERRUPTED)
        await self.state_mgr.transition_state(ConversationState.WAITING_FOR_INPUT)

        # Reset speculation flag
        self._speculation_started = False

        logger.debug(f"[{self.call_id}] Barge-in handled — ready for user input")

    # ------------------------------------------------------------------
    # Greeting (bypass LLM)
    # ------------------------------------------------------------------

    async def play_greeting(self) -> None:
        """
        Play the agent greeting without invoking LLM.

        Called once after call connects if agent has first_message.
        """
        agent = self.agent_config.get("agent")
        greeting_text = "Hello, how can I help you today?"

        if agent:
            first_message = getattr(agent, "first_message", None)
            if first_message:
                greeting_text = first_message

        self.tts_mgr.reset_for_new_turn()
        self.state_mgr.agent_speaking_start()

        tts_task = asyncio.create_task(
            self.tts_mgr.enqueue_chunk(
                text=greeting_text,
                cancellation_token=self.cancellation_token,
                is_final=True,
                is_quick_ack=False,
                end_call_after=False,
            ),
            name=f"greeting_tts_{self.call_id}",
        )
        await self.cancellation_token.register_task(tts_task)

        await self.state_mgr.add_to_history(role="agent", text=greeting_text)
        logger.info(f"[{self.call_id}] Greeting queued: '{greeting_text[:40]}'")

    # ------------------------------------------------------------------
    # Telemetry / inspection
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Return current state snapshot for telemetry / debugging."""
        return {
            **self.state_mgr.get_telemetry(),
            "speculation_started": self._speculation_started,
            "tts_speaking": self.tts_mgr.is_speaking,
            "llm_has_speculation": self.llm_mgr.has_speculation,
            "call_active": self._call_active,
        }

    def get_call_summary(self) -> Dict[str, Any]:
        """Return call summary for DB storage after call ends."""
        return self.state_mgr.get_call_summary()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _start_speculation(self, text: str) -> None:
        """
        Spawn LLM speculation as a non-blocking background task.

        Guards against double-starting per turn.
        """
        if self._speculation_started:
            return

        self._speculation_started = True
        self._llm_started_at_ms = int(time.monotonic() * 1000)

        self.tts_mgr.reset_for_new_turn()

        system_prompt = self._build_system_prompt()
        history = self.state_mgr.get_messages_for_llm()

        spec_task = asyncio.create_task(
            self.llm_mgr.stream_speculative(
                partial_text=text,
                cancellation_token=self.cancellation_token,
                conversation_history=history,
                system_prompt=system_prompt,
            ),
            name=f"llm_spec_{self.call_id}",
        )
        self.llm_mgr._speculative_task = spec_task
        await self.cancellation_token.register_task(spec_task)

        logger.info(
            f"[{self.call_id}] 🚀 LLM speculation started: '{text[:30]}...'"
        )

    def _build_system_prompt(self) -> str:
        """Build system prompt using agent config and conversation history."""
        agent = self.agent_config.get("agent")
        agent_name = "AI Assistant"
        agent_language = "en"

        if agent:
            agent_name = getattr(agent, "name", None) or "AI Assistant"
            agent_language = getattr(agent, "language", None) or "en"

        # Build history text from state manager
        history_text = self._build_history_text()

        return self.llm_mgr.build_system_prompt(
            agent_name=agent_name,
            agent_language=agent_language,
            history_text=history_text,
        )

    def _build_history_text(self) -> str:
        """Format conversation history for system prompt injection."""
        messages = self.state_mgr.get_messages_for_llm()
        if not messages:
            return ""

        lines = []
        for msg in messages:
            role_label = "Client" if msg.get("role") == "user" else "Agent"
            content = msg.get("content", "")
            if content:
                lines.append(f"{role_label}: {content}")

        return "\n".join(lines)

    def set_endpointing_mode(self, mode: str) -> None:
        """Adjust STT endpointing mode at runtime (e.g., for email collection)."""
        self.stt_mgr.set_endpointing_mode(mode)
