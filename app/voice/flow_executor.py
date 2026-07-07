"""Pure, CPU-bound traversal engine for pre-compiled visual call-flow graphs.

Consumes the ``flow_data_compiled`` JSONB produced by
``app.services.flow_graph_service.compile_graph`` — ``{node_id: {"node":
node_dict, "outgoing_edges": [sorted_edges]}}``. No I/O: callers are
responsible for actually speaking text, waiting for STT, transferring the
call, etc. Every node transition is budgeted at under 50ms; transitions
exceeding the budget are logged as warnings.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.logger import logger

NODE_TRANSITION_BUDGET_MS = 50.0

GREETING = "greeting"
COLLECT_INPUT = "collect_input"
BRANCH = "branch"
TRANSFER = "transfer"
END_CALL = "end_call"


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
    speech_text: Optional[str] = None


class FlowExecutorError(Exception):
    """Raised for malformed compiled graphs or unknown node types."""


class FlowExecutor:
    """Traverses a pre-compiled flow graph one node at a time."""

    def __init__(self, compiled_graph: Dict[str, Any]) -> None:
        self._graph = compiled_graph

    def start_node_id(self) -> str:
        """Return the id of the graph's single start node.

        A node is a start node if it has ``type == "start"`` or
        ``data.isStart``/``data.is_start`` truthy — mirrors
        ``flow_graph_service._is_start_node``.
        """
        for node_id, entry in self._graph.items():
            node = entry["node"]
            if node.get("type") == "start":
                return node_id
            data = node.get("data") or {}
            if data.get("isStart") or data.get("is_start"):
                return node_id
        raise FlowExecutorError("Compiled graph has no start node")

    def execute_node(self, node_id: str, state: PipelineState) -> NodeExecutionResult:
        """Run the node's action logic and record it in ``state.history``."""
        started = time.perf_counter()
        entry = self._graph.get(node_id)
        if entry is None:
            raise FlowExecutorError(f"Unknown node id: {node_id}")

        node = entry["node"]
        node_type = node.get("type")
        data = node.get("data") or {}

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
        else:
            raise FlowExecutorError(f"Unsupported node type: {node_type}")

        state.current_node_id = node_id
        state.history.append(node_id)
        self._log_timing("execute_node", node_id, started)
        return result

    def next_node_id(
        self,
        current_node_id: str,
        transcript: Optional[str],
        variables: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Evaluate outgoing edges from ``current_node_id`` in priority order.

        Returns the target node id of the first matching edge, falling back
        to a ``fallback`` edge if present, or ``None`` if nothing matches.
        """
        started = time.perf_counter()
        entry = self._graph.get(current_node_id)
        if entry is None:
            raise FlowExecutorError(f"Unknown node id: {current_node_id}")

        variables = variables or {}
        fallback_target: Optional[str] = None

        for edge in entry["outgoing_edges"]:
            condition = edge.get("condition") or {}
            condition_type = condition.get("type", "always")

            if condition_type == "fallback":
                if fallback_target is None:
                    fallback_target = edge.get("target")
                continue

            if self._condition_matches(
                condition_type, condition, transcript, variables
            ):
                self._log_timing("next_node_id", current_node_id, started)
                return edge.get("target")

        self._log_timing("next_node_id", current_node_id, started)
        return fallback_target

    def _condition_matches(
        self,
        condition_type: str,
        condition: Dict[str, Any],
        transcript: Optional[str],
        variables: Dict[str, Any],
    ) -> bool:
        if condition_type == "always":
            return True

        if transcript is None:
            return False

        if condition_type == "intent_match":
            pattern = condition.get("pattern")
            if not pattern:
                return False
            return re.search(pattern, transcript, re.IGNORECASE) is not None

        if condition_type == "keyword":
            keyword = condition.get("keyword")
            if not keyword:
                return False
            words = re.findall(r"\w+", transcript.lower())
            return keyword.lower() in words

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
