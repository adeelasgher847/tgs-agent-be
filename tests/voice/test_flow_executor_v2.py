"""Unit tests for FlowExecutor: entry point resolution, node execution, KB_LOOKUP,
and branch condition evaluation."""

from __future__ import annotations

import pytest

from app.voice.flow_executor import (
    BRANCH,
    COLLECT_INPUT,
    END_CALL,
    GREETING,
    KB_LOOKUP,
    TRANSFER,
    FlowExecutor,
    FlowExecutorError,
    NodeExecutionResult,
    PipelineState,
)


class TestFlowExecutorEntryPoint:
    def test_start_node_id_resolves_from_entry_node_id(self):
        graph = {
            "greet": {"type": GREETING, "data": {}, "next_nodes": {}},
        }
        executor = FlowExecutor(graph, entry_node_id="greet")

        assert executor.start_node_id() == "greet"

    def test_start_node_id_raises_when_entry_node_id_missing(self):
        executor = FlowExecutor({"greet": {"type": GREETING, "data": {}, "next_nodes": {}}})

        with pytest.raises(FlowExecutorError):
            executor.start_node_id()

    def test_start_node_id_raises_when_entry_node_id_not_in_graph(self):
        executor = FlowExecutor(
            {"greet": {"type": GREETING, "data": {}, "next_nodes": {}}},
            entry_node_id="does_not_exist",
        )

        with pytest.raises(FlowExecutorError):
            executor.start_node_id()


class TestFlowExecutorNodeExecution:
    def test_execute_node_kb_lookup(self):
        graph = {
            "kb_node": {
                "type": KB_LOOKUP,
                "data": {
                    "kb_id": "00000000-0000-0000-0000-000000000001",
                    "query_variable": "caller_question",
                    "result_variable": "policy_answer",
                    "min_confidence": 0.75,
                    "fallback_message": "Sorry, I could not find that policy.",
                },
                "next_nodes": {},
            }
        }
        executor = FlowExecutor(graph)
        state = PipelineState(current_node_id="kb_node")

        result = executor.execute_node("kb_node", state)

        assert isinstance(result, NodeExecutionResult)
        assert result.node_id == "kb_node"
        assert result.node_type == KB_LOOKUP
        assert result.action == "kb_lookup"
        assert result.config["result_variable"] == "policy_answer"
        assert result.config["min_confidence"] == 0.75
        assert state.history == ["kb_node"]

    def test_execute_node_unknown_type_raises(self):
        graph = {"weird": {"type": "not_a_real_type", "data": {}, "next_nodes": {}}}
        executor = FlowExecutor(graph)
        state = PipelineState(current_node_id="weird")

        with pytest.raises(FlowExecutorError):
            executor.execute_node("weird", state)


class TestFlowExecutorNonBranchNextNode:
    def test_greeting_follows_single_default_edge(self):
        graph = {
            "greet": {"type": GREETING, "data": {}, "next_nodes": {"default": "collect"}},
        }
        executor = FlowExecutor(graph, entry_node_id="greet")

        assert executor.next_node_id("greet", None, {}) == "collect"

    def test_transcript_content_never_affects_routing(self):
        graph = {
            "greet": {"type": GREETING, "data": {}, "next_nodes": {"default": "collect"}},
        }
        executor = FlowExecutor(graph, entry_node_id="greet")

        # Any transcript content is ignored — only the default handle is followed.
        assert executor.next_node_id("greet", "totally unrelated text", {}) == "collect"
        assert executor.next_node_id("greet", None, {}) == "collect"

    def test_terminal_node_with_no_next_nodes_returns_none(self):
        graph = {
            "end": {"type": END_CALL, "data": {}, "next_nodes": {}},
        }
        executor = FlowExecutor(graph)

        assert executor.next_node_id("end", None, {}) is None


class TestFlowExecutorBranchConditionEvaluation:
    @pytest.fixture
    def executor(self):
        return FlowExecutor({})

    def test_equals_string_match(self, executor):
        data = {"condition_variable": "reason", "operator": "equals", "condition_value": "billing"}
        assert executor._evaluate_branch_condition(data, {"reason": "billing"}) is True
        assert executor._evaluate_branch_condition(data, {"reason": "support"}) is False

    def test_equals_numeric_match(self, executor):
        data = {"condition_variable": "score", "operator": "equals", "condition_value": "10"}
        assert executor._evaluate_branch_condition(data, {"score": 10}) is True
        assert executor._evaluate_branch_condition(data, {"score": "10.0"}) is True
        assert executor._evaluate_branch_condition(data, {"score": 9}) is False

    def test_contains_operator(self, executor):
        data = {
            "condition_variable": "call_reason",
            "operator": "contains",
            "condition_value": "tow",
        }
        assert executor._evaluate_branch_condition(data, {"call_reason": "I need a tow truck"}) is True
        assert executor._evaluate_branch_condition(data, {"call_reason": "battery jump start"}) is False

    def test_greater_than_operator(self, executor):
        data = {
            "condition_variable": "interest_level",
            "operator": "greater_than",
            "condition_value": "6",
        }
        assert executor._evaluate_branch_condition(data, {"interest_level": "8"}) is True
        assert executor._evaluate_branch_condition(data, {"interest_level": 6}) is False
        assert executor._evaluate_branch_condition(data, {"interest_level": "3"}) is False
        # Non-numeric value fails gracefully
        assert executor._evaluate_branch_condition(data, {"interest_level": "very interested"}) is False

    def test_less_than_operator(self, executor):
        data = {"condition_variable": "balance", "operator": "less_than", "condition_value": "100"}
        assert executor._evaluate_branch_condition(data, {"balance": "50"}) is True
        assert executor._evaluate_branch_condition(data, {"balance": 100}) is False
        assert executor._evaluate_branch_condition(data, {"balance": "150"}) is False
        # Non-numeric value fails gracefully
        assert executor._evaluate_branch_condition(data, {"balance": "unknown"}) is False

    def test_is_empty_operator(self, executor):
        data = {"condition_variable": "api_result", "operator": "is_empty", "condition_value": ""}
        assert executor._evaluate_branch_condition(data, {"api_result": ""}) is True
        assert executor._evaluate_branch_condition(data, {"api_result": None}) is True
        assert executor._evaluate_branch_condition(data, {}) is True
        assert executor._evaluate_branch_condition(data, {"api_result": "found_record"}) is False

    def test_regex_match_operator(self, executor):
        data = {
            "condition_variable": "contract_number",
            "operator": "regex_match",
            "condition_value": r"^CNT-\d{5}$",
        }
        assert executor._evaluate_branch_condition(data, {"contract_number": "CNT-12345"}) is True
        assert executor._evaluate_branch_condition(data, {"contract_number": "INVALID"}) is False
        # Malformed regex pattern fails gracefully
        bad_data = {
            "condition_variable": "contract_number",
            "operator": "regex_match",
            "condition_value": "[unclosed",
        }
        assert executor._evaluate_branch_condition(bad_data, {"contract_number": "CNT-12345"}) is False

    def test_unknown_operator_returns_false(self, executor):
        data = {"condition_variable": "flag", "operator": "non_existent_op", "condition_value": "foo"}
        assert executor._evaluate_branch_condition(data, {"flag": "foo"}) is False

    def test_missing_variable_warning_and_empty_string_handling(self, executor, caplog):
        data = {
            "condition_variable": "missing_var",
            "operator": "equals",
            "condition_value": "expected_value",
        }
        assert executor._evaluate_branch_condition(data, {}) is False
        assert "variable 'missing_var' not found in state" in caplog.text


class TestFlowExecutorNextNodeIdBranchRouting:
    def test_next_node_id_follows_yes_or_no_handle(self):
        graph = {
            "branch_node": {
                "type": BRANCH,
                "data": {
                    "condition_variable": "consent",
                    "operator": "equals",
                    "condition_value": "yes",
                },
                "next_nodes": {"yes": "continue_flow", "no": "polite_exit"},
            }
        }
        executor = FlowExecutor(graph)

        assert executor.next_node_id("branch_node", None, {"consent": "yes"}) == "continue_flow"
        assert executor.next_node_id("branch_node", None, {"consent": "no"}) == "polite_exit"

    def test_next_node_id_missing_handle_returns_none(self):
        graph = {
            "branch_node": {
                "type": BRANCH,
                "data": {
                    "condition_variable": "consent",
                    "operator": "equals",
                    "condition_value": "yes",
                },
                # only "yes" wired up — a malformed graph missing the "no" edge
                "next_nodes": {"yes": "continue_flow"},
            }
        }
        executor = FlowExecutor(graph)

        assert executor.next_node_id("branch_node", None, {"consent": "no"}) is None
