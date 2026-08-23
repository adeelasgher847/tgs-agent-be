"""Unit tests for FlowPipelineMixin: variable interpolation, collect_input variable storage, and KB lookup."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.voice.flow_executor import (
    COLLECT_INPUT,
    GREETING,
    KB_LOOKUP,
    FlowExecutor,
    NodeExecutionResult,
    PipelineState,
)
from app.voice.flow_pipeline_mixin import FlowPipelineMixin, _interpolate_variables


class TestInterpolateVariables:
    def test_single_variable_substituted(self):
        text = "Hello {customer_name}, welcome!"
        vars_dict = {"customer_name": "Clifford"}
        assert _interpolate_variables(text, vars_dict) == "Hello Clifford, welcome!"

    def test_multiple_variables_substituted(self):
        text = "Hi {name}, your appointment is on {date} at {time}."
        vars_dict = {"name": "Alice", "date": "Monday", "time": "2pm"}
        assert _interpolate_variables(text, vars_dict) == "Hi Alice, your appointment is on Monday at 2pm."

    def test_missing_variable_replaced_with_empty_string_and_logs_warning(self, caplog):
        text = "Your code is {code}, {missing_greeting}"
        vars_dict = {"code": "1234"}
        result = _interpolate_variables(text, vars_dict)
        assert result == "Your code is 1234, "
        assert "FlowExecutor: variable 'missing_greeting' not found in state" in caplog.text

    def test_non_string_values_converted_to_str(self):
        text = "Score: {score}, Amount: ${amount}"
        vars_dict = {"score": 100, "amount": 49.99}
        assert _interpolate_variables(text, vars_dict) == "Score: 100, Amount: $49.99"

    def test_no_placeholders_unchanged(self):
        text = "No variables in this string."
        assert _interpolate_variables(text, {"a": "b"}) == "No variables in this string."

    def test_empty_string(self):
        assert _interpolate_variables("", {"a": "b"}) == ""


class DummyHandler(FlowPipelineMixin):
    def __init__(self):
        self._tts_pipeline = MagicMock()
        self._tts_pipeline.queue_tts = AsyncMock()
        self._use_ssml = False
        self._twilio_buffer_primed = False
        self.tenant_id = "00000000-0000-0000-0000-000000000001"
        self.agent_id = "00000000-0000-0000-0000-000000000002"
        self.transcripts_recorded = []

    async def _add_to_transcript(self, role, text, source):
        self.transcripts_recorded.append((role, text, source))

    async def _transfer_after_agent_request(self):
        pass

    async def _end_call_after_agent_request(self):
        pass


class TestFlowSpeakInterpolation:
    @pytest.mark.asyncio
    async def test_flow_speak_interpolates_and_queues_tts(self):
        handler = DummyHandler()
        state = PipelineState(current_node_id="n1", variables={"user": "Bob"})
        handler._flow_state = state

        result = NodeExecutionResult(
            node_id="n1",
            node_type=GREETING,
            action="speak",
            config={},
            speech_text="Welcome back, {user}!",
        )

        await handler._flow_speak(result)

        assert handler.transcripts_recorded == [("agent", "Welcome back, Bob!", "flow_node")]
        handler._tts_pipeline.queue_tts.assert_awaited_once_with(
            {
                "text": "Welcome back, Bob!",
                "chunk_id": "flow_n1",
                "use_ssml": False,
                "is_final": True,
            }
        )

    @pytest.mark.asyncio
    async def test_flow_speak_skips_tts_if_interpolated_text_empty(self):
        handler = DummyHandler()
        state = PipelineState(current_node_id="n1", variables={})
        handler._flow_state = state

        result = NodeExecutionResult(
            node_id="n1",
            node_type=GREETING,
            action="speak",
            config={},
            speech_text="{empty_missing}",
        )

        await handler._flow_speak(result)

        assert handler.transcripts_recorded == []
        handler._tts_pipeline.queue_tts.assert_not_called()


class TestFlowRunChainCollectInput:
    @pytest.mark.asyncio
    async def test_collect_input_stores_transcript_and_speaks_prompt(self):
        handler = DummyHandler()
        graph = {
            "greet": {
                "type": GREETING,
                "data": {},
                "next_nodes": {"default": "collect_node"},
            },
            "collect_node": {
                "type": COLLECT_INPUT,
                "data": {
                    "variable_name": "contract_number",
                    "prompt": "Please say your contract number, {customer_name}.",
                },
                "next_nodes": {},
            },
        }
        handler._flow_executor = FlowExecutor(graph, entry_node_id="greet")
        state = PipelineState(current_node_id="greet", variables={"customer_name": "Sarah"})
        handler._flow_state = state

        await handler._flow_run_chain(transcript="CNT-99887")

        # Variable stored from caller utterance
        assert state.variables["contract_number"] == "CNT-99887"
        # Prompt spoken with variable interpolation
        assert ("agent", "Please say your contract number, Sarah.", "flow_node") in handler.transcripts_recorded

    @pytest.mark.asyncio
    async def test_collect_input_does_not_store_none_transcript(self):
        handler = DummyHandler()
        graph = {
            "greet": {
                "type": GREETING,
                "data": {},
                "next_nodes": {"default": "collect_node"},
            },
            "collect_node": {
                "type": COLLECT_INPUT,
                "data": {"variable_name": "answer"},
                "next_nodes": {},
            },
        }
        handler._flow_executor = FlowExecutor(graph, entry_node_id="greet")
        state = PipelineState(current_node_id="greet", variables={"answer": "initial"})
        handler._flow_state = state

        await handler._flow_run_chain(transcript=None)

        assert state.variables["answer"] == "initial"


class TestFlowKbLookup:
    @pytest.mark.asyncio
    async def test_kb_lookup_empty_query_speaks_fallback(self):
        handler = DummyHandler()
        state = PipelineState(current_node_id="kb_node", variables={})
        handler._flow_state = state

        result = NodeExecutionResult(
            node_id="kb_node",
            node_type=KB_LOOKUP,
            action="kb_lookup",
            config={
                "query_variable": "caller_question",
                "result_variable": "policy_info",
                "fallback_message": "Could not find policy info.",
            },
        )

        with patch("app.services.rag_service.rag_service.retrieve") as mock_retrieve:
            await handler._flow_kb_lookup(result, state)
            mock_retrieve.assert_not_called()

        assert ("agent", "Could not find policy info.", "flow_node") in handler.transcripts_recorded
        assert "policy_info" not in state.variables

    @pytest.mark.asyncio
    async def test_kb_lookup_success_stores_highest_score_chunk(self):
        handler = DummyHandler()
        state = PipelineState(
            current_node_id="kb_node",
            variables={"caller_question": "What is the deductible?"},
        )
        handler._flow_state = state

        result = NodeExecutionResult(
            node_id="kb_node",
            node_type=KB_LOOKUP,
            action="kb_lookup",
            config={
                "query_variable": "caller_question",
                "result_variable": "kb_answer",
                "min_confidence": 0.70,
                "fallback_message": "No answer found.",
            },
        )

        chunk1 = MagicMock(score=0.85, text="Your deductible is $500.")
        chunk2 = MagicMock(score=0.72, text="Standard deductible applies.")

        with patch("app.services.rag_service.rag_service.retrieve", return_value=[chunk2, chunk1]):
            await handler._flow_kb_lookup(result, state)

        assert state.variables["kb_answer"] == "Your deductible is $500."

    @pytest.mark.asyncio
    async def test_kb_lookup_below_threshold_speaks_fallback(self):
        handler = DummyHandler()
        state = PipelineState(
            current_node_id="kb_node",
            variables={"caller_question": "What is covered?"},
        )
        handler._flow_state = state

        result = NodeExecutionResult(
            node_id="kb_node",
            node_type=KB_LOOKUP,
            action="kb_lookup",
            config={
                "query_variable": "caller_question",
                "result_variable": "kb_answer",
                "min_confidence": 0.80,
                "fallback_message": "I could not find information about that coverage.",
            },
        )

        low_chunk = MagicMock(score=0.65, text="Low relevance text")

        with patch("app.services.rag_service.rag_service.retrieve", return_value=[low_chunk]):
            await handler._flow_kb_lookup(result, state)

        assert "kb_answer" not in state.variables
        assert ("agent", "I could not find information about that coverage.", "flow_node") in handler.transcripts_recorded

    @pytest.mark.asyncio
    async def test_kb_lookup_exception_handled_gracefully(self):
        handler = DummyHandler()
        state = PipelineState(
            current_node_id="kb_node",
            variables={"caller_question": "Error trigger"},
        )
        handler._flow_state = state

        result = NodeExecutionResult(
            node_id="kb_node",
            node_type=KB_LOOKUP,
            action="kb_lookup",
            config={
                "query_variable": "caller_question",
                "result_variable": "kb_answer",
                "fallback_message": "Service unavailable right now.",
            },
        )

        with patch("app.services.rag_service.rag_service.retrieve", side_effect=RuntimeError("DB down")):
            await handler._flow_kb_lookup(result, state)

        assert "kb_answer" not in state.variables
        assert ("agent", "Service unavailable right now.", "flow_node") in handler.transcripts_recorded

    @pytest.mark.asyncio
    async def test_kb_lookup_passes_kb_id_through_to_retrieve(self):
        """A kb_lookup node with a configured kb_id must scope retrieve() to that KB."""
        import uuid

        handler = DummyHandler()
        state = PipelineState(
            current_node_id="kb_node",
            variables={"caller_question": "What is the returns policy?"},
        )
        handler._flow_state = state

        target_kb_id = "11111111-1111-1111-1111-111111111111"
        result = NodeExecutionResult(
            node_id="kb_node",
            node_type=KB_LOOKUP,
            action="kb_lookup",
            config={
                "kb_id": target_kb_id,
                "query_variable": "caller_question",
                "result_variable": "kb_answer",
                "min_confidence": 0.5,
                "fallback_message": "No answer found.",
            },
        )

        chunk = MagicMock(score=0.9, text="Returns are accepted within 30 days.")

        with patch(
            "app.services.rag_service.rag_service.retrieve", return_value=[chunk]
        ) as mock_retrieve:
            await handler._flow_kb_lookup(result, state)

        assert state.variables["kb_answer"] == "Returns are accepted within 30 days."
        _, kwargs = mock_retrieve.call_args
        assert kwargs["kb_id"] == uuid.UUID(target_kb_id)
        # Tenant scoping is preserved alongside kb_id scoping.
        assert str(kwargs["tenant_id"]) == handler.tenant_id

    @pytest.mark.asyncio
    async def test_kb_lookup_without_kb_id_retrieves_unscoped(self):
        """Backward compatibility: no kb_id configured means kb_id=None is passed through."""
        handler = DummyHandler()
        state = PipelineState(
            current_node_id="kb_node",
            variables={"caller_question": "What is the pricing?"},
        )
        handler._flow_state = state

        result = NodeExecutionResult(
            node_id="kb_node",
            node_type=KB_LOOKUP,
            action="kb_lookup",
            config={
                "query_variable": "caller_question",
                "result_variable": "kb_answer",
                "min_confidence": 0.5,
            },
        )

        chunk = MagicMock(score=0.9, text="Pricing starts at $99/mo.")

        with patch(
            "app.services.rag_service.rag_service.retrieve", return_value=[chunk]
        ) as mock_retrieve:
            await handler._flow_kb_lookup(result, state)

        _, kwargs = mock_retrieve.call_args
        assert kwargs["kb_id"] is None

    @pytest.mark.asyncio
    async def test_kb_lookup_invalid_kb_id_falls_back_to_unscoped(self):
        """A malformed kb_id must not crash the node; it degrades to unscoped retrieval."""
        handler = DummyHandler()
        state = PipelineState(
            current_node_id="kb_node",
            variables={"caller_question": "What is the pricing?"},
        )
        handler._flow_state = state

        result = NodeExecutionResult(
            node_id="kb_node",
            node_type=KB_LOOKUP,
            action="kb_lookup",
            config={
                "kb_id": "not-a-valid-uuid",
                "query_variable": "caller_question",
                "result_variable": "kb_answer",
                "min_confidence": 0.5,
            },
        )

        chunk = MagicMock(score=0.9, text="Pricing starts at $99/mo.")

        with patch(
            "app.services.rag_service.rag_service.retrieve", return_value=[chunk]
        ) as mock_retrieve:
            await handler._flow_kb_lookup(result, state)

        _, kwargs = mock_retrieve.call_args
        assert kwargs["kb_id"] is None
