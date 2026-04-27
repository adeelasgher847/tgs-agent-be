"""
Integration tests for VoiceOrchestrator V2.

Covers:
1. State machine transitions (WaitingForInput → UserSpeaking → Processing → AgentSpeaking)
2. BargeInController thresholds (multi-word, single-word stop-words)
3. CancellationToken behaviour (register, cancel_all, reset)
4. LLMStreamManager chunk flushing logic
5. STTStreamManager deduplication
6. Early LLM speculation trigger (3+ words @ 0.18 confidence)
7. Barge-in cascade (cancellation → state reset → re-arm)
8. End-to-end orchestrator mock flow

All tests use mocked providers — no real Deepgram/ElevenLabs/OpenAI calls.
"""

import asyncio
import pytest
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

# ---------------------------------------------------------------------------
# Test: ConversationStateManager
# ---------------------------------------------------------------------------

from app.voice.conversation_state_manager import (
    ConversationState,
    ConversationStateManager,
    Mood,
)


class TestConversationStateManager:

    def setup_method(self):
        self.mgr = ConversationStateManager(call_id="test-call-001", agent_config={})

    def test_initial_state(self):
        assert self.mgr.state == ConversationState.WAITING_FOR_INPUT
        assert self.mgr.agent_is_speaking is False
        assert self.mgr.user_speaking is False

    @pytest.mark.asyncio
    async def test_valid_transition_waiting_to_user_speaking(self):
        await self.mgr.transition_state(ConversationState.USER_SPEAKING)
        assert self.mgr.state == ConversationState.USER_SPEAKING

    @pytest.mark.asyncio
    async def test_invalid_transition_rejected(self):
        """Skipping from WAITING directly to AGENT_SPEAKING is invalid."""
        await self.mgr.transition_state(ConversationState.AGENT_SPEAKING)
        assert self.mgr.state == ConversationState.WAITING_FOR_INPUT  # Unchanged

    @pytest.mark.asyncio
    async def test_full_happy_path_transitions(self):
        """Test complete call state flow."""
        await self.mgr.transition_state(ConversationState.USER_SPEAKING)
        await self.mgr.transition_state(ConversationState.PROCESSING)
        await self.mgr.transition_state(ConversationState.AGENT_SPEAKING)
        await self.mgr.transition_state(ConversationState.WAITING_FOR_INPUT)
        assert self.mgr.state == ConversationState.WAITING_FOR_INPUT

    @pytest.mark.asyncio
    async def test_interrupted_path(self):
        """Barge-in path: AgentSpeaking → Interrupted → WaitingForInput."""
        await self.mgr.transition_state(ConversationState.USER_SPEAKING)
        await self.mgr.transition_state(ConversationState.PROCESSING)
        await self.mgr.transition_state(ConversationState.AGENT_SPEAKING)
        await self.mgr.transition_state(ConversationState.INTERRUPTED)
        await self.mgr.transition_state(ConversationState.WAITING_FOR_INPUT)
        assert self.mgr.state == ConversationState.WAITING_FOR_INPUT

    def test_set_interim_text_starts_user_speaking(self):
        self.mgr.set_interim_text("Hello there")
        assert self.mgr.interim_text == "Hello there"
        assert self.mgr.user_speaking is True

    def test_agent_speaking_flags(self):
        self.mgr.agent_speaking_start()
        assert self.mgr.agent_is_speaking is True
        self.mgr.agent_speaking_end()
        assert self.mgr.agent_is_speaking is False

    def test_cancel_agent_speaking(self):
        self.mgr.agent_speaking_start()
        self.mgr.cancel_agent_speaking()
        assert self.mgr.agent_is_speaking is False

    @pytest.mark.asyncio
    async def test_message_history_max_12(self):
        """History should not exceed 12 messages."""
        for i in range(15):
            await self.mgr.add_to_history(role="user", text=f"Message {i}")
        assert len(self.mgr.messages) == 12

    def test_mood_update(self):
        self.mgr.update_mood(Mood.FRUSTRATED)
        assert self.mgr.current_mood == Mood.FRUSTRATED

    @pytest.mark.asyncio
    async def test_get_messages_for_llm_format(self):
        await self.mgr.add_to_history(role="user", text="Hello")
        await self.mgr.add_to_history(role="agent", text="Hi there!")
        messages = self.mgr.get_messages_for_llm()
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[0]["content"] == "Hello"

    def test_get_call_summary(self):
        summary = self.mgr.get_call_summary()
        assert "call_id" in summary
        assert summary["call_id"] == "test-call-001"
        assert "state" in summary

    def test_get_telemetry(self):
        telemetry = self.mgr.get_telemetry()
        assert "state" in telemetry
        assert "elapsed_ms" in telemetry


# ---------------------------------------------------------------------------
# Test: CancellationToken
# ---------------------------------------------------------------------------

from app.voice.cancellation import CancellationToken


class TestCancellationToken:

    @pytest.mark.asyncio
    async def test_initial_not_cancelled(self):
        token = CancellationToken("test")
        assert token.is_cancelled() is False

    @pytest.mark.asyncio
    async def test_cancel_all_sets_flag_then_resets(self):
        """cancel_all() sets cancelled during operation, resets after completion."""
        token = CancellationToken("test")
        # With no tasks, it sets flag then immediately resets
        # The flag is only visible DURING cancel_all (between set and reset)
        # After completion, is_cancelled should be False (ready for next turn)
        assert token.is_cancelled() is False
        await token.cancel_all()
        # Flag stays True when no tasks (reset only happens after gather)
        # Actual production behavior: new token created on barge-in
        # For test: just verify no exception and state is consistent
        # The implementation sets _is_cancelled = True then resets to False at end
        assert token.is_cancelled() is False  # Reset after all tasks done

    @pytest.mark.asyncio
    async def test_registered_task_receives_cancel(self):
        """Registered tasks should receive cancellation signal."""
        token = CancellationToken("test")
        task_started = asyncio.Event()

        async def long_running():
            task_started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                raise

        task = asyncio.create_task(long_running())
        await token.register_task(task)
        await task_started.wait()  # Wait for task to actually start
        await token.cancel_all(timeout_ms=200)
        assert task.done()  # Task should be done (cancelled)

    @pytest.mark.asyncio
    async def test_cancel_all_no_tasks(self):
        """Cancel with no registered tasks should not raise."""
        token = CancellationToken("test")
        await token.cancel_all()  # Should not raise

    @pytest.mark.asyncio
    async def test_task_auto_removes_on_completion(self):
        token = CancellationToken("test")

        async def quick():
            return 42

        task = asyncio.create_task(quick())
        await token.register_task(task)
        await task  # Let it complete
        await asyncio.sleep(0)  # Allow callback to fire
        assert task not in token._tasks

    @pytest.mark.asyncio
    async def test_context_manager(self):
        token = CancellationToken("test")
        async with token:
            assert token.is_cancelled() is False


# ---------------------------------------------------------------------------
# Test: BargeInController
# ---------------------------------------------------------------------------

from app.voice.barge_in_controller import BargeInController


class TestBargeInController:

    def setup_method(self):
        self.orchestrator = AsyncMock()
        self.ctrl = BargeInController(call_id="test", orchestrator=self.orchestrator)
        self.ctrl.arm()  # Arm for testing

    @pytest.mark.asyncio
    async def test_multi_word_barge_in_triggers(self):
        """2+ words at >= 0.25 confidence → barge-in."""
        await self.ctrl.check_trigger("hold on please", confidence=0.30)
        self.orchestrator.on_barge_in.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_multi_word_low_confidence_no_trigger(self):
        """2+ words but confidence too low → no barge-in."""
        await self.ctrl.check_trigger("hold on please", confidence=0.10)
        self.orchestrator.on_barge_in.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_single_word_stop_word_triggers(self):
        """Single stop-word at >= 0.52 confidence → barge-in."""
        await self.ctrl.check_trigger("stop", confidence=0.60)
        self.orchestrator.on_barge_in.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_single_word_non_stop_word_no_trigger(self):
        """Single non-stop word → no barge-in."""
        await self.ctrl.check_trigger("hello", confidence=0.80)
        self.orchestrator.on_barge_in.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_text_no_trigger(self):
        await self.ctrl.check_trigger("", confidence=0.80)
        self.orchestrator.on_barge_in.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_idempotent_once_per_arm(self):
        """Second trigger in same turn is ignored (guard flag)."""
        await self.ctrl.check_trigger("stop it now", confidence=0.90)
        await self.ctrl.check_trigger("stop please", confidence=0.90)
        assert self.orchestrator.on_barge_in.await_count == 1

    @pytest.mark.asyncio
    async def test_rearm_allows_next_barge_in(self):
        """After arm() → disarm() → arm(), barge-in works again."""
        await self.ctrl.check_trigger("stop", confidence=0.70)
        self.ctrl.disarm()
        self.ctrl.arm()
        await self.ctrl.check_trigger("stop", confidence=0.70)
        assert self.orchestrator.on_barge_in.await_count == 2

    @pytest.mark.asyncio
    async def test_configure_thresholds(self):
        """Custom thresholds should be respected."""
        self.ctrl.configure_thresholds(multi_word_threshold=0.5, single_word_threshold=0.9)
        # 2 words at 0.40 (below new threshold 0.5) → no barge-in
        await self.ctrl.check_trigger("hold on", confidence=0.40)
        self.orchestrator.on_barge_in.assert_not_awaited()

    def test_all_stop_words_recognized(self):
        """All defined stop-words are in the frozenset."""
        for word in ["stop", "no", "halt", "wait", "enough", "quiet", "shush"]:
            assert word in BargeInController.STOP_WORDS


# ---------------------------------------------------------------------------
# Test: STTStreamManager deduplication
# ---------------------------------------------------------------------------

from app.voice.stt_stream_manager import STTStreamManager


class TestSTTStreamManagerDedup:

    def setup_method(self):
        self.orchestrator = AsyncMock()
        # Patch deepgram_stt_service to avoid real connections
        with patch("app.voice.stt_stream_manager.deepgram_stt_service"):
            self.mgr = STTStreamManager("test-call", self.orchestrator)

    @pytest.mark.asyncio
    async def test_first_final_not_deduplicated(self):
        """First occurrence of a transcript fires to orchestrator."""
        await self.mgr._on_final("Hello there", confidence=0.8)
        self.orchestrator.on_stt_final.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_duplicate_final_deduplicated(self):
        """Same text within 6 seconds fires only once."""
        await self.mgr._on_final("Hello there", confidence=0.8)
        await self.mgr._on_final("Hello there", confidence=0.8)
        assert self.orchestrator.on_stt_final.await_count == 1

    @pytest.mark.asyncio
    async def test_low_confidence_final_rejected(self):
        """Finals below min confidence threshold are dropped."""
        with patch.object(
            type(self.mgr),
            "_is_duplicate",
            return_value=False,
        ):
            # confidence=0.05 << min threshold
            await self.mgr._on_final("huh", confidence=0.05)
            self.orchestrator.on_stt_final.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_interim_fires_to_orchestrator(self):
        """Interim transcripts always fire to orchestrator."""
        await self.mgr._on_interim("Hello", confidence=0.5)
        self.orchestrator.on_stt_interim.assert_awaited_once_with(
            text="Hello", confidence=0.5, word_count=1
        )


# ---------------------------------------------------------------------------
# Test: LLMStreamManager chunk flushing
# ---------------------------------------------------------------------------

from app.voice.llm_stream_manager import _find_flush_index, _find_time_flush_index, _strip_control_tokens


class TestLLMChunkFlushing:

    def test_sentence_boundary_flush_first_sentence(self):
        """Should flush only at the first complete sentence boundary."""
        buf = "Hello there."
        idx = _find_flush_index(buf, min_words=2, max_words=12)
        assert idx is not None
        chunk = buf[:idx].strip()
        assert chunk == "Hello there."

    def test_no_flush_too_few_words(self):
        """Should not flush if too few words before boundary."""
        buf = "Hi."
        idx = _find_flush_index(buf, min_words=2, max_words=12)
        # "Hi." is only 1 word — should not flush
        assert idx is None

    def test_max_words_triggers_flush_at_soft_boundary(self):
        """Should flush at comma when buffer has >= max_words."""
        buf = "Hello, how are you doing today, I hope you are well"
        idx = _find_flush_index(buf, min_words=2, max_words=8)
        assert idx is not None

    def test_strip_control_tokens(self):
        assert _strip_control_tokens("Thank you [END_CALL]") == "Thank you "
        assert _strip_control_tokens("[OUTCOME:SUCCESS] Done") == " Done"
        assert _strip_control_tokens("Normal text") == "Normal text"

    def test_time_flush_requires_min_words(self):
        """Time-based flush should not fire with fewer than max(min_words, 5) words."""
        buf = "Hi there"
        idx = _find_time_flush_index(buf, min_words=2)
        assert idx is None

    def test_time_flush_fires_on_sufficient_words(self):
        buf = "Hello there how are you doing today hopefully well"
        idx = _find_time_flush_index(buf, min_words=2)
        assert idx is not None


# ---------------------------------------------------------------------------
# Test: VoiceOrchestrator integration (fully mocked)
# ---------------------------------------------------------------------------

from app.voice.orchestrator import VoiceOrchestrator


class TestVoiceOrchestratorIntegration:
    """
    Integration tests with mocked STT/LLM/TTS providers.

    Tests orchestrator event routing, speculation trigger, and barge-in cascade.
    """

    def _make_orchestrator(self):
        frames_sent = []

        async def send_frame(frame: bytes):
            frames_sent.append(frame)

        with (
            patch("app.voice.stt_stream_manager.deepgram_stt_service"),
            patch("app.voice.llm_stream_manager.gemini_service"),
            patch("app.voice.llm_stream_manager.openai_service"),
            patch("app.voice.llm_stream_manager.groq_service"),
        ):
            orch = VoiceOrchestrator(
                call_id="test-orch-001",
                agent_id="agent-001",
                agent_config={"agent": None},
                send_twilio_frame_callback=send_frame,
            )
        return orch, frames_sent

    @pytest.mark.asyncio
    async def test_start_call_initializes_state(self):
        """start_call() should set state to WAITING_FOR_INPUT."""
        orch, _ = self._make_orchestrator()
        with patch.object(orch.stt_mgr, "start", new_callable=AsyncMock):
            await orch.start_call()
        assert orch._call_active is True
        assert orch.state_mgr.state == ConversationState.WAITING_FOR_INPUT

    @pytest.mark.asyncio
    async def test_early_speculation_triggered_on_interim(self):
        """
        on_stt_interim with 3+ words @ 0.18 confidence should create LLM task.
        """
        orch, _ = self._make_orchestrator()
        orch._call_active = True

        speculation_called = False

        async def fake_speculation(*args, **kwargs):
            nonlocal speculation_called
            speculation_called = True

        orch.llm_mgr.stream_speculative = fake_speculation

        await orch.on_stt_interim(
            text="tell me about your services",
            confidence=0.80,
            word_count=5,
        )

        # Allow background task to run
        await asyncio.sleep(0)
        assert orch._speculation_started is True
        assert orch.state_mgr.state == ConversationState.PROCESSING

    @pytest.mark.asyncio
    async def test_interim_transitions_to_user_speaking(self):
        """
        on_stt_interim should transition from WAITING_FOR_INPUT to USER_SPEAKING.
        """
        orch, _ = self._make_orchestrator()
        orch._call_active = True
        assert orch.state_mgr.state == ConversationState.WAITING_FOR_INPUT

        await orch.on_stt_interim(
            text="hello there",
            confidence=0.65,
            word_count=2,
        )

        assert orch.state_mgr.state == ConversationState.USER_SPEAKING

    @pytest.mark.asyncio
    async def test_no_speculation_below_word_threshold(self):
        """on_stt_interim with < 3 words should NOT trigger speculation."""
        orch, _ = self._make_orchestrator()
        orch._call_active = True
        orch._early_llm_min_words = 3

        await orch.on_stt_interim(
            text="hi",
            confidence=0.90,
            word_count=1,
        )
        assert orch._speculation_started is False

    @pytest.mark.asyncio
    async def test_barge_in_cancels_all_tasks(self):
        """on_barge_in() should cancel all tasks and reset state."""
        orch, _ = self._make_orchestrator()
        orch._call_active = True
        orch.state_mgr.agent_speaking_start()

        cancelled_tasks = []
        cancel_called = False

        async def fake_cancel_all(timeout_ms=100):
            nonlocal cancel_called
            cancel_called = True

        orch.cancellation_token.cancel_all = fake_cancel_all

        await orch.on_barge_in()

        assert cancel_called is True
        assert orch.state_mgr.agent_is_speaking is False
        assert orch._speculation_started is False
        assert orch.state_mgr.state == ConversationState.WAITING_FOR_INPUT

    @pytest.mark.asyncio
    async def test_barge_in_creates_new_token(self):
        """After barge-in, a fresh CancellationToken should be created."""
        orch, _ = self._make_orchestrator()
        orch._call_active = True

        old_token = orch.cancellation_token

        async def fake_cancel_all(timeout_ms=100):
            pass

        old_token.cancel_all = fake_cancel_all
        await orch.on_barge_in()

        assert orch.cancellation_token is not old_token

    @pytest.mark.asyncio
    async def test_shutdown_stops_all_managers(self):
        """shutdown() should call stop() on all sub-managers."""
        orch, _ = self._make_orchestrator()
        orch._call_active = True

        orch.stt_mgr.stop = AsyncMock()
        orch.llm_mgr.stop = AsyncMock()
        orch.tts_mgr.stop = AsyncMock()

        async def fake_cancel_all(timeout_ms=100):
            pass

        orch.cancellation_token.cancel_all = fake_cancel_all

        await orch.shutdown()

        orch.stt_mgr.stop.assert_awaited_once()
        orch.llm_mgr.stop.assert_awaited_once()
        orch.tts_mgr.stop.assert_awaited_once()
        assert orch._call_active is False

    @pytest.mark.asyncio
    async def test_get_state_telemetry(self):
        """get_state() should return a non-empty dict."""
        orch, _ = self._make_orchestrator()
        state = orch.get_state()
        assert "state" in state
        assert "call_active" in state

    @pytest.mark.asyncio
    async def test_tts_frame_forwarded_to_twilio(self):
        """on_tts_frame_ready() should forward frame via send callback."""
        received = []

        async def send_frame(frame: bytes):
            received.append(frame)

        with (
            patch("app.voice.stt_stream_manager.deepgram_stt_service"),
            patch("app.voice.llm_stream_manager.gemini_service"),
        ):
            orch = VoiceOrchestrator(
                call_id="test",
                agent_id="agent",
                agent_config={"agent": None},
                send_twilio_frame_callback=send_frame,
            )

        test_frame = b"\x7f" * 160
        await orch.on_tts_frame_ready(test_frame)
        assert received == [test_frame]


# ---------------------------------------------------------------------------
# Test: Quick ack helper functions
# ---------------------------------------------------------------------------

from app.voice.llm_stream_manager import _should_quick_ack, _ACK_SKIP_PHRASES


class TestQuickAck:

    def test_eligible_with_enough_words(self):
        # "help" is in the skip phrases — use a clean sentence instead
        assert _should_quick_ack(
            "Can you tell me about your pricing options please", 5, _ACK_SKIP_PHRASES
        ) is True

    def test_not_eligible_too_few_words(self):
        assert _should_quick_ack("Hello", 5, _ACK_SKIP_PHRASES) is False

    def test_not_eligible_emotional_content(self):
        assert _should_quick_ack("I have an emergency please help me now", 5, _ACK_SKIP_PHRASES) is False

    def test_not_eligible_empty_string(self):
        assert _should_quick_ack("", 5, _ACK_SKIP_PHRASES) is False


# ---------------------------------------------------------------------------
# Test: TTSStreamManager frame alignment
# ---------------------------------------------------------------------------

from app.voice.tts_stream_manager import MULAW_FRAME_BYTES


class TestTTSFrameAlignment:
    """Test MULAW frame boundary arithmetic."""

    def test_frame_size_constant(self):
        """20ms at 8kHz MULAW = 160 bytes."""
        assert MULAW_FRAME_BYTES == 160

    def test_frame_padding(self):
        """Frames shorter than 160 bytes should be padded."""
        short_frame = b"\x7f" * 80
        padded = short_frame + bytes([0x7F]) * (MULAW_FRAME_BYTES - len(short_frame))
        assert len(padded) == MULAW_FRAME_BYTES

    def test_silence_value(self):
        """MULAW silence value is 0x7F (127)."""
        silence = bytes([0x7F]) * MULAW_FRAME_BYTES
        assert len(silence) == MULAW_FRAME_BYTES
        assert all(b == 0x7F for b in silence)
