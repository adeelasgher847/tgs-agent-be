"""
LiveKit/browser counterpart of tests/voice/test_quick_ack_gating.py.

Before this change, `ConversationOrchestrator.send_quick_acknowledgement`
had NONE of Twilio's cooldown/variety/content gates — it was a bare
`random.choice()` on every eligible turn. This regressed the transport
parity requirement (humanization behavior must not diverge between Twilio
and Browser/LiveKit): the same conversation on the browser path could say
"Got it" every single turn, or repeat the exact same phrase back-to-back,
neither of which Twilio allowed.

`ConversationOrchestrator.send_quick_acknowledgement` now delegates to the
shared `decide_quick_ack` (the SAME function Twilio calls), so these tests
mirror the Twilio cooldown/variety/content gate tests using the same
probability-forcing fixture pattern.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.voice import conversation_orchestrator as co_mod
from app.voice.conversation_orchestrator import ConversationOrchestrator

SUBSTANTIVE_TEXT = "I would like to schedule an appointment for next Tuesday afternoon please"


def _base_orchestrator() -> ConversationOrchestrator:
    handler = MagicMock()
    handler._tts_pipeline = MagicMock()
    handler._tts_pipeline.queue_tts = AsyncMock()
    return ConversationOrchestrator(handler)


@pytest.fixture(autouse=True)
def _force_quick_ack_enabled():
    """Force the probability roll to always pass, mirroring the Twilio
    suite's fixture — VOICE_TUNABLES is the same shared singleton both
    transports read."""
    original = co_mod.VOICE_TUNABLES.quick_ack.probability
    object.__setattr__(co_mod.VOICE_TUNABLES.quick_ack, "probability", 1.0)
    yield
    object.__setattr__(co_mod.VOICE_TUNABLES.quick_ack, "probability", original)


class TestContentGate:
    @pytest.mark.asyncio
    async def test_short_backchannel_never_acked(self):
        orch = _base_orchestrator()
        orch._quick_ack_turns_since_last = co_mod.QUICK_ACK_COOLDOWN_TURNS
        await orch.send_quick_acknowledgement("okay")
        orch._h._tts_pipeline.queue_tts.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_text_is_noop(self):
        orch = _base_orchestrator()
        orch._quick_ack_turns_since_last = co_mod.QUICK_ACK_COOLDOWN_TURNS
        await orch.send_quick_acknowledgement("   ")
        orch._h._tts_pipeline.queue_tts.assert_not_awaited()


class TestCooldownGate:
    @pytest.mark.asyncio
    async def test_ack_withheld_until_cooldown_elapses(self):
        orch = _base_orchestrator()
        assert orch._quick_ack_turns_since_last == 0

        for _ in range(co_mod.QUICK_ACK_COOLDOWN_TURNS - 1):
            await orch.send_quick_acknowledgement(SUBSTANTIVE_TEXT)
            orch._h._tts_pipeline.queue_tts.assert_not_awaited()

        await orch.send_quick_acknowledgement(SUBSTANTIVE_TEXT)
        orch._h._tts_pipeline.queue_tts.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_counter_resets_after_ack_fires(self):
        orch = _base_orchestrator()
        orch._quick_ack_turns_since_last = co_mod.QUICK_ACK_COOLDOWN_TURNS

        await orch.send_quick_acknowledgement(SUBSTANTIVE_TEXT)
        assert orch._h._tts_pipeline.queue_tts.await_count == 1
        assert orch._quick_ack_turns_since_last == 0

        # Immediately eligible again, but cooldown must be re-earned.
        await orch.send_quick_acknowledgement(SUBSTANTIVE_TEXT)
        assert orch._h._tts_pipeline.queue_tts.await_count == 1


class TestVarietyGate:
    @pytest.mark.asyncio
    async def test_second_consecutive_ack_uses_different_phrase(self):
        orch = _base_orchestrator()
        orch._quick_ack_turns_since_last = co_mod.QUICK_ACK_COOLDOWN_TURNS

        await orch.send_quick_acknowledgement(SUBSTANTIVE_TEXT)
        first_phrase = orch._h._tts_pipeline.queue_tts.await_args_list[0].args[0]["text"]

        orch._quick_ack_turns_since_last = co_mod.QUICK_ACK_COOLDOWN_TURNS
        await orch.send_quick_acknowledgement(SUBSTANTIVE_TEXT)
        second_phrase = orch._h._tts_pipeline.queue_tts.await_args_list[1].args[0]["text"]

        assert first_phrase != second_phrase


class TestSharedDecisionFunction:
    def test_livekit_and_twilio_use_the_identical_shared_function(self):
        import app.routers.bidirectional_stream as twilio_mod

        assert twilio_mod.decide_quick_ack is co_mod.decide_quick_ack
