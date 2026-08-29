"""
Regression test for the lead-capture "confirmation loop" defect: with a
tenant custom (or model) system prompt driving a "collect name/phone/email/
address" flow, the LLM would alternate indefinitely between re-asking for
already-acknowledged fields (e.g. "Just to confirm, what is your email
address?" / "Just to confirm, what is your full address?") because:

  1. The Twilio path's (bidirectional_stream.py) anti-repeat instruction
     lived only in the advisory "CRITICAL RULES" block, which is rendered
     AFTER the tenant's "CUSTOM INSTRUCTIONS"/"MODEL INSTRUCTIONS" and is
     never declared to override them (unlike "GROUNDING RULES", which is
     explicitly non-negotiable and rendered BEFORE custom/model content).
  2. The LiveKit path's (conversation_orchestrator.py) custom/model branches
     had no explicit "don't re-ask/re-confirm already-acknowledged fields"
     rule at all (only a generic "don't repeat questions" line), and no
     GROUNDING RULES coverage either.

Fix: add an explicit, override-authoritative "NO CONFIRMATION LOOPS" rule to
GROUNDING RULES in both custom and model branches of both handlers, and
strengthen/backfill the CRITICAL RULES conversation-continuity language to
explicitly call out "Got it"-style acknowledgement and never alternating
between two already-answered fields.

Only prompt-text assembly is asserted (mirrors
test_bracket_tag_prompt_consistency.py's approach) — LLM compliance itself
is untestable in this codebase.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from app.routers.bidirectional_stream import BidirectionalStreamHandler
from app.voice.conversation_orchestrator import ConversationOrchestrator
from tests.voice.test_call_pipeline import _base_handler
from tests.voice.test_bracket_tag_prompt_consistency import (
    _capture_twilio_system_prompt,
    _fake_livekit_handler,
)

GROUNDING_MARKER = "NO CONFIRMATION LOOPS"
CONTINUITY_MARKER = "CONVERSATION CONTINUITY"


def _run_twilio(h: BidirectionalStreamHandler, spy_stream):
    with patch(
        "app.routers.bidirectional_stream.openai_service.stream_text", new=spy_stream
    ), patch.object(h, "_add_to_transcript", new=AsyncMock()), patch.object(
        h, "_send_in_progress_status", new=AsyncMock()
    ):
        asyncio.run(h.generate_and_stream_response("Hello there", 0.9))


class TestTwilioNoConfirmationLoopGuardrail:
    def test_custom_system_prompt_has_grounding_rule_and_continuity_language(self):
        h = _base_handler()
        h.agent.system_prompt = "Collect the caller's name, phone, email, and address."
        h.agent.model.system_prompt = None
        h.agent.tts_provider.slug = "google"
        captured, spy = _capture_twilio_system_prompt(h)
        _run_twilio(h, spy)

        sp = captured[0]
        assert GROUNDING_MARKER in sp, "GROUNDING RULES must carry an explicit anti-loop rule"
        # The grounding rule must appear BEFORE the tenant's custom instructions
        # so it is structurally override-authoritative, not merely advisory.
        assert sp.index(GROUNDING_MARKER) < sp.index("# CUSTOM INSTRUCTIONS")
        assert '"Got it"' in sp

    def test_model_system_prompt_has_grounding_rule_and_continuity_language(self):
        h = _base_handler()
        h.agent.system_prompt = None
        h.agent.model.system_prompt = "Follow the recruiting script."
        h.agent.tts_provider.slug = "google"
        captured, spy = _capture_twilio_system_prompt(h)
        _run_twilio(h, spy)

        sp = captured[0]
        assert GROUNDING_MARKER in sp
        assert sp.index(GROUNDING_MARKER) < sp.index("# MODEL INSTRUCTIONS")
        assert '"Got it"' in sp


class TestLiveKitNoConfirmationLoopGuardrail:
    def test_custom_system_prompt_has_grounding_rule_and_continuity_language(self):
        h = _fake_livekit_handler(
            tts_slug="google",
            custom_system_prompt="Collect the caller's name, phone, email, and address.",
        )
        orchestrator = ConversationOrchestrator(h)
        sp = asyncio.run(orchestrator.build_system_prompt("Hello there", 0.9))

        assert GROUNDING_MARKER in sp
        assert sp.index(GROUNDING_MARKER) < sp.index("# CUSTOM INSTRUCTIONS")
        assert CONTINUITY_MARKER in sp
        assert '"Got it"' in sp

    def test_model_system_prompt_has_grounding_rule_and_continuity_language(self):
        h = _fake_livekit_handler(
            tts_slug="google",
            model_system_prompt="Follow the recruiting script.",
        )
        orchestrator = ConversationOrchestrator(h)
        sp = asyncio.run(orchestrator.build_system_prompt("Hello there", 0.9))

        assert GROUNDING_MARKER in sp
        assert sp.index(GROUNDING_MARKER) < sp.index("# MODEL INSTRUCTIONS")
        assert CONTINUITY_MARKER in sp
        assert '"Got it"' in sp
