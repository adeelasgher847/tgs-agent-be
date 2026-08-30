"""
Regression coverage: Twilio's three system-prompt branches (base agent
prompt, agent.system_prompt custom override, agent.model.system_prompt
override) must each instruct the LLM to (1) speak phone numbers/confirmation
codes/digit sequences in a naturally-grouped, comma-separated spoken form
instead of one fast unbroken string, and (2) add a brief acknowledgement
beat after delivering a specific requested detail (phone number, email,
etc.) before moving on to the next question -- rather than jumping straight
into it with no pause.

Root cause (from a reviewed real call recording): neither instruction
existed anywhere in the prompt, so the LLM had no guidance to slow down or
add spoken-form punctuation for digit sequences, and no instruction that a
short acknowledgement is appropriate right after delivering a requested
detail -- both read back too fast / transitioned too abruptly as a direct
consequence.

Fix is text-only (system prompt content), reusing the exact real-handler
fixture and system-prompt-capture pattern already proven in
tests/voice/test_bracket_tag_prompt_consistency.py -- no audio pipeline,
STT, or TTS-timing changes, so this cannot introduce latency and does not
touch app/voice/livekit_browser_call_handler.py or
app/voice/conversation_orchestrator.py (the demo/LiveKit path builds its
system prompt via an entirely separate implementation, untouched here).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from app.routers.bidirectional_stream import BidirectionalStreamHandler
from tests.voice.test_call_pipeline import _base_handler

STRUCTURED_INFO_MARKER = "STRUCTURED INFO PACING"
TRANSITION_PACING_MARKER = "TRANSITION PACING"


def _capture_twilio_system_prompt(h: BidirectionalStreamHandler) -> tuple[list, callable]:
    captured = []

    async def _spy_stream(prompt=None, system_prompt=None, **kwargs):
        captured.append(system_prompt or "")
        yield "Sure, here is a short reply."

    return captured, _spy_stream


def _run(h, spy_stream):
    with patch(
        "app.routers.bidirectional_stream.openai_service.stream_text", new=spy_stream
    ), patch.object(h, "_add_to_transcript", new=AsyncMock()), patch.object(
        h, "_send_in_progress_status", new=AsyncMock()
    ):
        asyncio.run(h.generate_and_stream_response("What's your phone number?", 0.9))


class TestStructuredInfoPacingPresentInAllTwilioPromptBranches:
    def test_base_prompt_includes_both_pacing_instructions(self):
        h = _base_handler()
        h.agent.system_prompt = None
        h.agent.model.system_prompt = None
        captured, spy = _capture_twilio_system_prompt(h)
        _run(h, spy)

        sp = captured[0]
        assert STRUCTURED_INFO_MARKER in sp
        assert TRANSITION_PACING_MARKER in sp

    def test_custom_agent_system_prompt_includes_both_pacing_instructions(self):
        h = _base_handler()
        h.agent.system_prompt = "Custom instructions for this agent."
        h.agent.model.system_prompt = None
        captured, spy = _capture_twilio_system_prompt(h)
        _run(h, spy)

        sp = captured[0]
        assert STRUCTURED_INFO_MARKER in sp
        assert TRANSITION_PACING_MARKER in sp

    def test_model_system_prompt_includes_both_pacing_instructions(self):
        h = _base_handler()
        h.agent.system_prompt = None
        h.agent.model.system_prompt = "Follow the recruiting script."
        captured, spy = _capture_twilio_system_prompt(h)
        _run(h, spy)

        sp = captured[0]
        assert STRUCTURED_INFO_MARKER in sp
        assert TRANSITION_PACING_MARKER in sp

    def test_structured_info_instruction_gives_comma_grouped_example(self):
        """Guards the specific technique (comma-grouped spoken digits), not
        just presence of the marker word -- this is the concrete mechanism
        research confirmed actually works with this provider (punctuation-
        driven pauses), as opposed to SSML break tags which are unreliable
        for the ElevenLabs model this deployment uses."""
        h = _base_handler()
        h.agent.system_prompt = None
        h.agent.model.system_prompt = None
        captured, spy = _capture_twilio_system_prompt(h)
        _run(h, spy)

        sp = captured[0]
        assert "five five five, one two three, four five six seven" in sp

    def test_livekit_conversation_orchestrator_prompt_is_untouched(self):
        """Explicit scope guard: the demo/LiveKit path's separate prompt
        builder must not gain this Twilio-specific instruction text."""
        import inspect

        from app.voice import conversation_orchestrator

        source = inspect.getsource(conversation_orchestrator)
        assert STRUCTURED_INFO_MARKER not in source
        assert TRANSITION_PACING_MARKER not in source
