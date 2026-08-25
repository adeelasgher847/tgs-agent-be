"""Coverage for the System Webhooks `{{key}}` prompt/greeting injection call
sites added to app/voice/conversation_orchestrator.py (LiveKit browser calls)
and app/routers/bidirectional_stream.py (Twilio calls).

ConversationOrchestrator.build_system_prompt / generate_and_stream_response
already have an established, directly-testable fixture pattern in
tests/voice/test_build_system_prompt_extraction.py and
tests/voice/test_bracket_tag_prompt_consistency.py (`_fake_livekit_handler`),
so those two injection sites are exercised end-to-end below, including the
regression-safety "absent webhook_variables leaves text unchanged" case.

BidirectionalStreamHandler's equivalent sites (`_build_system_prompt_full`,
`generate_and_stream_response`'s greeting block) are NOT exercised directly
here: no existing test file constructs enough of that handler's dependency
graph (agent_service, RAG/KB retrieval, LLM-runtime resolution, quick-ack
task, A/B prompt override, etc. — see the ~500-line method starting at
app/routers/bidirectional_stream.py:1551) to reach the injection block at
the end without a large, brittle amount of new mocking that would test the
mock more than the code. Per the test-writer agent's brief, that gap is
flagged explicitly here rather than forced — a reasonable follow-up would be
extracting the two 6-line injection blocks in bidirectional_stream.py into a
small shared helper (mirroring how `apply_field_mapping_values` is already
reused across both call sites) so it can be unit-tested in isolation without
constructing the whole handler; that's a production-code change outside this
agent's scope, so it is reported rather than made.

Instead, the two `render_template` unit tests below at least confirm the
*utility* behaves correctly against a payload shape that matches what
`app/routers/voice.py::handle_incoming_call` actually writes onto
`call_session.call_metadata["webhook_variables"]` (see
tests/routers/test_voice_dynamic_inbound_routing.py::
TestDynamicInboundCallRouting::test_webhook_variables_and_call_flow_id_attached_to_session
for where that shape is asserted end-to-end at the write side) — i.e. proof
that if `bidirectional_stream.py`'s call sites pass call_metadata's
webhook_variables into render_template exactly as written, the values come
out rendered correctly. It does not prove the call sites do so (that part is
the flagged gap).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from app.utils.webhook_templating import render_template
from app.voice.conversation_orchestrator import ConversationOrchestrator
from tests.voice.test_bracket_tag_prompt_consistency import _fake_livekit_handler


class TestConversationOrchestratorSystemPromptInjection:
    def test_webhook_variables_injected_into_system_prompt(self, monkeypatch):
        h = _fake_livekit_handler(
            tts_slug="google",
            custom_system_prompt="Hello {{customer_name}}, thanks for calling {{_metadata.company}}.",
        )
        h.call_session = SimpleNamespace(
            call_transcript=None,
            call_metadata={
                "webhook_variables": {
                    "customer_name": "Alice",
                    "_metadata": {"company": "Acme"},
                }
            },
        )
        orchestrator = ConversationOrchestrator(h)

        prompt = asyncio.run(orchestrator.build_system_prompt("Hi", 0.9))

        assert "Hello Alice, thanks for calling Acme." in prompt

    def test_absent_webhook_variables_leaves_prompt_unchanged(self, monkeypatch):
        """Regression safety: the common case (no Pre-Inbound webhook
        configured) must be a complete no-op — the raw {{token}} (if any)
        passes through untouched rather than being rendered against an
        empty/wrong context."""
        h = _fake_livekit_handler(
            tts_slug="google",
            custom_system_prompt="Hello {{customer_name}}, thanks for calling.",
        )
        # _fake_livekit_handler's default call_session is a falsy
        # _FakeCallSession() — the exact "no webhook configured" shape.
        orchestrator = ConversationOrchestrator(h)

        prompt = asyncio.run(orchestrator.build_system_prompt("Hi", 0.9))

        assert "Hello {{customer_name}}, thanks for calling." in prompt

    def test_webhook_injection_failure_does_not_break_prompt_building(
        self, monkeypatch
    ):
        """The injection block is wrapped in its own try/except. Since
        render_template() is itself designed to never raise (see
        tests/utils/test_webhook_templating.py), the only realistic way to
        exercise this try/except is to force render_template to raise —
        confirming the rest of prompt building still completes, using the
        pre-injection system_prompt, rather than propagating."""
        h = _fake_livekit_handler(
            tts_slug="google",
            custom_system_prompt="Hello {{customer_name}}.",
        )
        h.call_session = SimpleNamespace(
            call_transcript=None,
            call_metadata={"webhook_variables": {"customer_name": "Alice"}},
        )
        orchestrator = ConversationOrchestrator(h)

        with patch(
            "app.voice.conversation_orchestrator.render_template",
            side_effect=RuntimeError("simulated templating bug"),
        ):
            prompt = asyncio.run(orchestrator.build_system_prompt("Hi", 0.9))

        # Injection failed silently -> pre-injection (raw-token) text preserved.
        assert "Hello {{customer_name}}." in prompt


class TestConversationOrchestratorGreetingInjection:
    def _run_greeting(self, h) -> str:
        orchestrator = ConversationOrchestrator(h)
        asyncio.run(
            orchestrator.generate_and_stream_response("", 0.9, is_greeting=True)
        )
        assert h._tts_pipeline.queue_tts.await_count == 1
        return h._tts_pipeline.queue_tts.await_args.args[0]["text"]

    def test_webhook_variables_injected_into_greeting(self, monkeypatch):
        h = _fake_livekit_handler(tts_slug="google")
        h.agent.greeting_message = "Hi {{customer_name}}, welcome back!"
        h._voice_orchestrator = None  # bypass Gemini/OpenAI Realtime hand-off branches
        h.call_session = SimpleNamespace(
            call_transcript=None,
            call_metadata={"webhook_variables": {"customer_name": "Bob"}},
        )

        greeting = self._run_greeting(h)

        assert greeting == "Hi Bob, welcome back!"

    def test_absent_webhook_variables_leaves_greeting_unchanged(self, monkeypatch):
        h = _fake_livekit_handler(tts_slug="google")
        h.agent.greeting_message = "Hi {{customer_name}}, welcome back!"
        h._voice_orchestrator = None
        # Default _FakeCallSession() is falsy -> {} webhook_variables.

        greeting = self._run_greeting(h)

        assert greeting == "Hi {{customer_name}}, welcome back!"


class TestRenderTemplateAgainstRealCallMetadataShape:
    """Confirms render_template's behavior against the exact
    call_metadata["webhook_variables"] shape app/routers/voice.py's
    Dynamic Inbound Call Routing wiring actually writes (flat str->str
    dict) — see tests/routers/test_voice_dynamic_inbound_routing.py for
    where that write path itself is covered."""

    def test_renders_prompt_and_greeting_text_from_real_webhook_variables_shape(self):
        webhook_variables = {"account_tier": "gold", "agent_id": "irrelevant-here"}

        system_prompt = "You are assisting a {{account_tier}} tier customer."
        greeting = "Hi there! I see you're a {{account_tier}} member."

        assert (
            render_template(system_prompt, webhook_variables)
            == "You are assisting a gold tier customer."
        )
        assert (
            render_template(greeting, webhook_variables)
            == "Hi there! I see you're a gold member."
        )

    def test_missing_variable_in_real_shape_renders_empty_not_raise(self):
        webhook_variables = {"account_tier": "gold"}
        text = "Reference: {{ticket_id}}"

        assert render_template(text, webhook_variables) == "Reference: "
