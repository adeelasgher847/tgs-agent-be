"""
Coverage for the quick-acknowledgement gating rewrite in
`BidirectionalStreamHandler._send_quick_acknowledgement` (branch
fix/quick-ack-turn-gating).

Previously, quick-ack dedup compared normalized transcript text within a 12s
window, which failed to prevent duplicate "Got it"s across the
interim-speculative call and the STT-final call for the same logical turn
(`generate_and_stream_response` can invoke `_send_quick_acknowledgement`
independently from both `_maybe_process_interim` and
`_complete_llm_turn_after_stt_final`).

The fix replaces that with four ordered gates, all state on `self`:
  1. Turn-identity (`_turn_generation_id` / `_last_quick_ack_turn_id`) —
     exactly one eligibility evaluation per logical turn.
  2. Content (`is_known_non_actionable_backchannel`) — never ack the
     caller's own short backchannel/confirmation.
  3. Cooldown (`_turns_since_last_ack` vs `QUICK_ACK_COOLDOWN_TURNS`) — only
     roll the probability check every Nth eligible turn.
  4. Variety (`_last_quick_ack_phrase`) — don't repeat the same phrase
     back-to-back.

`_cancel_inflight_llm_response` resets `_last_quick_ack_turn_id` (barge-in)
but never touches `_turns_since_last_ack`.

The whole method is a no-op end-to-end while
`settings.VOICE_QUICK_ACK_PROBABILITY <= 0.0` (default-disabled feature
flag), which is checked BEFORE `VOICE_TUNABLES.quick_ack.probability` is
ever consulted for the actual probability roll -- so tests must force both:
`settings.VOICE_QUICK_ACK_PROBABILITY` (the top-level enable/disable flag,
read live via `getattr`) AND `VOICE_TUNABLES.quick_ack.probability` (a
frozen dataclass field captured at import time, so it must be patched via
`object.__setattr__`, not through `settings`).
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import settings
from app.routers import bidirectional_stream as bds
from app.routers.bidirectional_stream import BidirectionalStreamHandler as Handler

# A clearly substantive, non-backchannel utterance with enough words to pass
# both `should_send_quick_ack`'s min-word gate and the content gate.
SUBSTANTIVE_TEXT = "I would like to schedule an appointment for next Tuesday afternoon please"


def _base_handler() -> Handler:
    """Minimal Handler instance via object.__new__, mirroring the pattern in
    tests/voice/test_call_pipeline.py — only sets attributes needed to
    exercise `_send_quick_acknowledgement` / `_cancel_inflight_llm_response`.
    """
    h = object.__new__(Handler)

    h.call_session = MagicMock()
    h.call_session.id = uuid.uuid4()
    h.call_session.tenant_id = uuid.uuid4()

    h._tts_pipeline = MagicMock()
    h._tts_pipeline.queue_tts = AsyncMock()
    h._tts_pipeline.cancel_current_and_clear_queue = AsyncMock()

    h._llm_response_task = None
    h._interim_task_response_produced = False
    h._current_speaking_agent_text = ""

    # New quick-ack gate state (see class docstring above).
    h._turn_generation_id = 0
    h._last_quick_ack_turn_id = -1
    h._turns_since_last_ack = 0
    h._last_quick_ack_phrase = ""

    return h


@pytest.fixture(autouse=True)
def _force_quick_ack_enabled(monkeypatch):
    """Force the feature flag on and the probability roll to always pass,
    for tests that need to get past the gates. Restored automatically by
    monkeypatch after each test.
    """
    monkeypatch.setattr(settings, "VOICE_QUICK_ACK_PROBABILITY", 1.0, raising=False)
    # `VOICE_TUNABLES.quick_ack` is a frozen dataclass captured at import
    # time in app.voice.conversation_orchestrator — patch its `probability`
    # field directly so `random.random() >= probability` is always False
    # (real random.random() is always in [0, 1), so probability=1.0 always
    # passes deterministically without needing to mock random itself).
    object.__setattr__(bds.VOICE_TUNABLES.quick_ack, "probability", 1.0)
    yield


class TestFeatureFlagDisabled:
    @pytest.mark.asyncio
    async def test_noop_when_probability_zero(self, monkeypatch):
        h = _base_handler()
        monkeypatch.setattr(settings, "VOICE_QUICK_ACK_PROBABILITY", 0.0, raising=False)
        h._turns_since_last_ack = Handler.QUICK_ACK_COOLDOWN_TURNS
        await h._send_quick_acknowledgement(SUBSTANTIVE_TEXT)
        h._tts_pipeline.queue_tts.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_noop_when_probability_unset(self, monkeypatch):
        h = _base_handler()
        monkeypatch.delattr(settings, "VOICE_QUICK_ACK_PROBABILITY", raising=False)
        h._turns_since_last_ack = Handler.QUICK_ACK_COOLDOWN_TURNS
        await h._send_quick_acknowledgement(SUBSTANTIVE_TEXT)
        h._tts_pipeline.queue_tts.assert_not_awaited()


class TestTurnIdentityGate:
    @pytest.mark.asyncio
    async def test_second_call_for_same_turn_id_is_noop(self):
        """Simulates the interim call then the final call for one logical turn
        (both share the same `_turn_generation_id`) -- only the first call's
        worth of ack evaluation should have any effect."""
        h = _base_handler()
        h._turns_since_last_ack = Handler.QUICK_ACK_COOLDOWN_TURNS

        # First call (e.g. speculative interim) for turn id 0.
        await h._send_quick_acknowledgement(SUBSTANTIVE_TEXT)
        assert h._tts_pipeline.queue_tts.await_count == 1

        # Second call (e.g. STT-final regeneration) for the SAME turn id,
        # different text -- must be a pure no-op regardless of content.
        await h._send_quick_acknowledgement("a totally different substantive sentence here")
        assert h._tts_pipeline.queue_tts.await_count == 1

    @pytest.mark.asyncio
    async def test_new_turn_id_is_eligible_again(self):
        h = _base_handler()
        h._turns_since_last_ack = Handler.QUICK_ACK_COOLDOWN_TURNS

        await h._send_quick_acknowledgement(SUBSTANTIVE_TEXT)
        assert h._tts_pipeline.queue_tts.await_count == 1

        # Same turn id again -- blocked.
        await h._send_quick_acknowledgement(SUBSTANTIVE_TEXT)
        assert h._tts_pipeline.queue_tts.await_count == 1

        # New logical turn (simulates _maybe_process_interim /
        # _complete_llm_turn_after_stt_final incrementing the counter) and
        # cooldown satisfied again -- eligible for a fresh ack.
        h._turn_generation_id += 1
        h._turns_since_last_ack = Handler.QUICK_ACK_COOLDOWN_TURNS
        await h._send_quick_acknowledgement(SUBSTANTIVE_TEXT)
        assert h._tts_pipeline.queue_tts.await_count == 2


class TestInterimTooShortFinalQualifies:
    @pytest.mark.asyncio
    async def test_short_interim_does_not_permanently_block_final_ack(self):
        """Regression for a real gap: VOICE_MIN_INTERIM_WORDS (4) is lower
        than QUICK_ACK_MIN_WORDS (5), so a speculative interim can reach
        `_send_quick_acknowledgement` with text too short for
        `should_send_quick_ack`, while the STT-final call for the SAME
        logical turn (same turn_id -- no new interim start, per
        `_complete_llm_turn_after_stt_final`) has the complete, longer text
        that WOULD qualify. The final call must still get a fair shot."""
        h = _base_handler()
        h._turns_since_last_ack = Handler.QUICK_ACK_COOLDOWN_TURNS

        # Interim: 4 words, below should_send_quick_ack's 5-word minimum.
        await h._send_quick_acknowledgement("I need an appointment")
        h._tts_pipeline.queue_tts.assert_not_awaited()

        # Final: same turn_id, full/complete text -- must still be eligible.
        await h._send_quick_acknowledgement(SUBSTANTIVE_TEXT)
        h._tts_pipeline.queue_tts.assert_awaited_once()


class TestContentGate:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("backchannel_text", ["okay", "yes", "yeah yeah", "hi there"])
    async def test_backchannel_never_acked(self, backchannel_text):
        h = _base_handler()
        h._turns_since_last_ack = Handler.QUICK_ACK_COOLDOWN_TURNS
        await h._send_quick_acknowledgement(backchannel_text)
        h._tts_pipeline.queue_tts.assert_not_awaited()
        # Turn-identity gate only locks on an actual send (see
        # TestTurnIdentityGate), so a content-gate rejection must NOT lock
        # the turn -- a later call for the same turn_id (e.g. the final,
        # with different/fuller text) still gets a fair evaluation.
        assert h._last_quick_ack_turn_id == -1


class TestCooldownGate:
    @pytest.mark.asyncio
    async def test_no_ack_below_cooldown_threshold(self):
        h = _base_handler()
        assert Handler.QUICK_ACK_COOLDOWN_TURNS == 3
        h._turns_since_last_ack = 0
        await h._send_quick_acknowledgement(SUBSTANTIVE_TEXT)
        h._tts_pipeline.queue_tts.assert_not_awaited()
        # Passing gates 1+2 and should_send_quick_ack increments the counter
        # even though the cooldown itself blocks the ack.
        assert h._turns_since_last_ack == 1

    @pytest.mark.asyncio
    async def test_ack_fires_once_cooldown_elapsed_and_resets(self):
        h = _base_handler()
        h._turns_since_last_ack = 0

        # Turns 1 and 2: eligible content, but below cooldown threshold (3).
        await h._send_quick_acknowledgement(SUBSTANTIVE_TEXT)
        h._turn_generation_id += 1
        await h._send_quick_acknowledgement(SUBSTANTIVE_TEXT)
        h._tts_pipeline.queue_tts.assert_not_awaited()
        assert h._turns_since_last_ack == 2

        # Turn 3: reaches the threshold -- ack fires and cooldown resets.
        h._turn_generation_id += 1
        await h._send_quick_acknowledgement(SUBSTANTIVE_TEXT)
        h._tts_pipeline.queue_tts.assert_awaited_once()
        assert h._turns_since_last_ack == 0

    @pytest.mark.asyncio
    async def test_backchannel_turn_does_not_advance_cooldown(self):
        h = _base_handler()
        h._turns_since_last_ack = 0
        await h._send_quick_acknowledgement("okay")
        # Content gate rejected before the cooldown counter is touched.
        assert h._turns_since_last_ack == 0

    @pytest.mark.asyncio
    async def test_duplicate_turn_id_after_ack_sent_does_not_advance_cooldown(self):
        """Once a turn has actually produced an ack, the counterpart call for
        the SAME turn_id (e.g. final, after interim already acked) must be a
        pure no-op -- it should not re-touch the cooldown counter at all."""
        h = _base_handler()
        h._turns_since_last_ack = Handler.QUICK_ACK_COOLDOWN_TURNS
        await h._send_quick_acknowledgement(SUBSTANTIVE_TEXT)
        h._tts_pipeline.queue_tts.assert_awaited_once()
        assert h._turns_since_last_ack == 0

        # Same turn id, different text -- blocked by gate 1 before the
        # cooldown counter is touched again.
        await h._send_quick_acknowledgement("a totally different substantive sentence here")
        h._tts_pipeline.queue_tts.assert_awaited_once()
        assert h._turns_since_last_ack == 0

    @pytest.mark.asyncio
    async def test_duplicate_turn_id_without_prior_ack_still_advances_cooldown(self):
        """If the first call for a turn_id didn't actually send an ack (e.g.
        cooldown not yet elapsed), the turn is NOT locked, so a counterpart
        call for the same turn_id gets its own fair cooldown evaluation --
        this is the accepted tradeoff of only locking on actual send (see
        `_send_quick_acknowledgement`'s Gate 1 docstring): a turn that takes
        two calls to resolve may count twice toward the cooldown, but can
        never produce two acks."""
        h = _base_handler()
        h._turns_since_last_ack = 0
        await h._send_quick_acknowledgement(SUBSTANTIVE_TEXT)
        h._tts_pipeline.queue_tts.assert_not_awaited()
        assert h._turns_since_last_ack == 1

        await h._send_quick_acknowledgement(SUBSTANTIVE_TEXT)
        h._tts_pipeline.queue_tts.assert_not_awaited()
        assert h._turns_since_last_ack == 2


class TestVarietyGate:
    @pytest.mark.asyncio
    async def test_second_consecutive_ack_uses_different_phrase(self):
        h = _base_handler()
        h._turns_since_last_ack = Handler.QUICK_ACK_COOLDOWN_TURNS

        await h._send_quick_acknowledgement(SUBSTANTIVE_TEXT)
        first_call_kwargs = h._tts_pipeline.queue_tts.await_args_list[0].args[0]
        first_phrase = first_call_kwargs["text"]
        assert h._last_quick_ack_phrase == first_phrase

        # New logical turn, cooldown satisfied again.
        h._turn_generation_id += 1
        h._turns_since_last_ack = Handler.QUICK_ACK_COOLDOWN_TURNS
        await h._send_quick_acknowledgement(SUBSTANTIVE_TEXT)
        second_call_kwargs = h._tts_pipeline.queue_tts.await_args_list[1].args[0]
        second_phrase = second_call_kwargs["text"]

        assert second_phrase != first_phrase
        assert h._last_quick_ack_phrase == second_phrase


class TestCancellationResetsTurnIdentityOnly:
    @pytest.mark.asyncio
    async def test_cancel_resets_turn_id_but_not_cooldown(self):
        h = _base_handler()
        h._last_quick_ack_turn_id = 5
        h._turns_since_last_ack = 2

        await h._cancel_inflight_llm_response()

        assert h._last_quick_ack_turn_id == -1
        assert h._turns_since_last_ack == 2
        h._tts_pipeline.cancel_current_and_clear_queue.assert_awaited_once()
