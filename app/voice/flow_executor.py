"""Pure, CPU-bound traversal engine for pre-compiled visual call-flow graphs.

Consumes the ``compiled_plan`` JSONB produced by
``app.services.flow_graph_service.compile_graph`` — ``{node_id: {"type": str,
"data": dict, "next_nodes": {handle: target_node_id}}}``. No I/O: callers are
responsible for actually speaking text, waiting for STT, transferring the
call, etc. Every node transition is budgeted at under 50ms; transitions
exceeding the budget are logged as warnings.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from app.core.logger import logger

NODE_TRANSITION_BUDGET_MS = 50.0

GREETING = "greeting"
COLLECT_INPUT = "collect_input"
BRANCH = "branch"
TRANSFER = "transfer"
END_CALL = "end_call"
KB_LOOKUP = "kb_lookup"

DEFAULT_HANDLE = "default"
YES_HANDLE = "yes"
NO_HANDLE = "no"


@dataclass
class PipelineState:
    """Per-call runtime state for a flow-driven conversation."""

    current_node_id: str
    history: List[str] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeExecutionResult:
    """Outcome of executing a single node — the caller acts on ``action``."""

    node_id: str
    node_type: str
    action: str
    config: Dict[str, Any]
    speech_text: str | None = None


class FlowExecutorError(Exception):
    """Raised for malformed compiled graphs or unknown node types."""


class FlowExecutor:
    """Traverses a pre-compiled flow graph one node at a time.

    ``entry_node_id`` is the flow document's own ``entry_node_id`` field
    (always the flow's single ``greeting`` node) — the executor never infers
    the entry point from node flags, per the ticket's literal entry-point
    mechanism.
    """

    def __init__(
        self, compiled_graph: Dict[str, Any], entry_node_id: str | None = None
    ) -> None:
        self._graph = compiled_graph
        self._entry_node_id = entry_node_id

    def start_node_id(self) -> str:
        """Return the flow's declared entry node id.

        Raises ``FlowExecutorError`` if no entry node id was supplied or it
        doesn't reference a node present in the compiled graph.
        """
        if not self._entry_node_id or self._entry_node_id not in self._graph:
            raise FlowExecutorError(
                f"Compiled graph has no usable entry node (entry_node_id={self._entry_node_id!r})"
            )
        return self._entry_node_id

    def execute_node(self, node_id: str, state: PipelineState) -> NodeExecutionResult:
        """Run the node's action logic and record it in ``state.history``."""
        started = time.perf_counter()
        entry = self._graph.get(node_id)
        if entry is None:
            raise FlowExecutorError(f"Unknown node id: {node_id}")

        node_type = entry.get("type")
        data = entry.get("data") or {}

        if node_type == GREETING:
            result = NodeExecutionResult(
                node_id=node_id,
                node_type=node_type,
                action="speak",
                config=data,
                speech_text=data.get("message"),
            )
        elif node_type == COLLECT_INPUT:
            result = NodeExecutionResult(
                node_id=node_id,
                node_type=node_type,
                action="wait_for_input",
                config=data,
            )
        elif node_type == BRANCH:
            result = NodeExecutionResult(
                node_id=node_id, node_type=node_type, action="branch", config=data
            )
        elif node_type == TRANSFER:
            result = NodeExecutionResult(
                node_id=node_id, node_type=node_type, action="transfer", config=data
            )
        elif node_type == END_CALL:
            result = NodeExecutionResult(
                node_id=node_id, node_type=node_type, action="end_call", config=data
            )
        elif node_type == KB_LOOKUP:
            result = NodeExecutionResult(
                node_id=node_id, node_type=node_type, action="kb_lookup", config=data
            )
        else:
            raise FlowExecutorError(f"Unsupported node type: {node_type}")

        state.current_node_id = node_id
        state.history.append(node_id)
        self._log_timing("execute_node", node_id, started)
        return result

    def next_node_id(
        self,
        current_node_id: str,
        transcript: str | None,
        variables: Dict[str, Any] | None = None,
    ) -> str | None:
        """Select the next node id per the ticket's per-node-type executor logic.

        ``branch`` nodes evaluate their own ``condition_variable``/``operator``
        /``condition_value`` (from the node's ``data``) against ``variables``
        and follow the ``yes`` or ``no`` handle. Every other node type follows
        its single ``default`` handle. ``transcript`` is accepted for call-site
        symmetry with the STT turn loop but does not affect routing — no node
        type in the ticket spec routes by transcript content.
        """
        started = time.perf_counter()
        entry = self._graph.get(current_node_id)
        if entry is None:
            raise FlowExecutorError(f"Unknown node id: {current_node_id}")

        variables = variables or {}
        next_nodes = entry.get("next_nodes") or {}

        if entry.get("type") == BRANCH:
            matched = self._evaluate_branch_condition(entry.get("data") or {}, variables)
            handle = YES_HANDLE if matched else NO_HANDLE
            target = next_nodes.get(handle)
            self._log_timing("next_node_id", current_node_id, started)
            return target

        target = next_nodes.get(DEFAULT_HANDLE)
        self._log_timing("next_node_id", current_node_id, started)
        return target

    def _evaluate_branch_condition(
        self, data: Dict[str, Any], variables: Dict[str, Any]
    ) -> bool:
        """Evaluate a ``branch`` node's condition against ``variables``.

        Reads ``condition_variable``/``operator``/``condition_value`` directly
        from the branch node's own config, per the ticket's BRANCH row in the
        node executor logic table. Supported operators: ``equals``,
        ``contains``, ``greater_than``, ``less_than``, ``is_empty``,
        ``regex_match``.
        """
        condition_variable = data.get("condition_variable", "")
        operator = data.get("operator", "equals")
        condition_value = data.get("condition_value", "")

        if condition_variable not in variables:
            logger.warning(
                "FlowExecutor: variable '%s' not found in state for branch condition, treating as empty",
                condition_variable,
            )
        raw_value = variables.get(condition_variable, "")

        if operator == "is_empty":
            return raw_value == "" or raw_value is None
        elif operator == "equals":
            try:
                return float(raw_value) == float(condition_value)
            except (ValueError, TypeError):
                return str(raw_value) == str(condition_value)
        elif operator == "contains":
            return str(condition_value) in str(raw_value)
        elif operator == "greater_than":
            try:
                return float(raw_value) > float(condition_value)
            except (ValueError, TypeError):
                logger.warning(
                    "FlowExecutor: branch greater_than: '%s' (value %r) is non-numeric; treating as false",
                    condition_variable,
                    raw_value,
                )
                return False
        elif operator == "less_than":
            try:
                return float(raw_value) < float(condition_value)
            except (ValueError, TypeError):
                logger.warning(
                    "FlowExecutor: branch less_than: '%s' (value %r) is non-numeric; treating as false",
                    condition_variable,
                    raw_value,
                )
                return False
        elif operator == "regex_match":
            try:
                return re.search(str(condition_value), str(raw_value)) is not None
            except re.error as exc:
                logger.warning(
                    "FlowExecutor: branch regex_match: invalid pattern %r: %s",
                    condition_value,
                    exc,
                )
                return False
        else:
            logger.warning(
                "FlowExecutor: unknown branch operator %r; treating as false", operator
            )
            return False

    def _log_timing(self, op: str, node_id: str, started: float) -> None:
        elapsed_ms = (time.perf_counter() - started) * 1000
        if elapsed_ms > NODE_TRANSITION_BUDGET_MS:
            logger.warning(
                "FlowExecutor.%s exceeded %sms budget for node %s: %.3fms",
                op,
                NODE_TRANSITION_BUDGET_MS,
                node_id,
                elapsed_ms,
            )
        else:
            logger.debug(
                "FlowExecutor.%s for node %s took %.3fms", op, node_id, elapsed_ms
            )
