"""Visual Flow Editor graph validation and pre-compilation.

Pure, CPU-bound functions operating on the raw React-Flow-shaped
``{"nodes": [...], "edges": [...], "entry_node_id": str}`` payload — no DB/IO.
Node shape: ``{"id": str, "type": str, "data": dict}``. Edge shape:
``{"id": str, "source": str, "target": str, "sourceHandle": str | None}``.

Per the BE6-S6-01 ticket spec: every flow has exactly one ``greeting`` node,
which is always the entry point, referenced by the flow document's top-level
``entry_node_id`` field — there is no separate "start" node type.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Nodes that leave the flow entirely (hang up or hand off) — exempt from the
# "must have an outgoing edge" rule.
TERMINAL_NODE_TYPES = {"end_call", "transfer"}

ENTRY_NODE_TYPE = "greeting"

# Edge sourceHandle used when a raw edge doesn't specify one.
DEFAULT_HANDLE = "default"


def validate_graph(flow_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate a flow graph. Returns a list of error dicts (empty if valid).

    Checks: exactly one ``greeting`` node, a valid ``entry_node_id`` pointing
    at that node, no directed cycles (DFS + recursion stack, O(V+E)), no
    orphan nodes (unreachable from the entry node), and every non-terminal
    node has at least one outgoing edge.
    """
    errors: List[Dict[str, Any]] = []
    nodes = flow_data.get("nodes") or []
    edges = flow_data.get("edges") or []

    node_by_id: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        node_id = node.get("id")
        if node_id is None:
            errors.append(
                {
                    "code": "invalid_node",
                    "message": "Node is missing an 'id'",
                    "node_id": None,
                }
            )
            continue
        node_by_id[node_id] = node

    if not node_by_id:
        errors.append(
            {
                "code": "empty_graph",
                "message": "Flow must contain at least one node",
                "node_id": None,
            }
        )
        return errors

    greeting_nodes = [
        n for n in node_by_id.values() if n.get("type") == ENTRY_NODE_TYPE
    ]
    if len(greeting_nodes) == 0:
        errors.append(
            {
                "code": "no_greeting_node",
                "message": "Flow must contain exactly one greeting node",
                "node_id": None,
            }
        )
    elif len(greeting_nodes) > 1:
        errors.append(
            {
                "code": "multiple_greeting_nodes",
                "message": f"Flow must contain exactly one greeting node, found {len(greeting_nodes)}",
                "node_id": None,
            }
        )

    entry_node_id = flow_data.get("entry_node_id")
    entry_node = node_by_id.get(entry_node_id) if entry_node_id else None
    if not entry_node_id:
        errors.append(
            {
                "code": "missing_entry_node_id",
                "message": "Flow must declare an 'entry_node_id'",
                "node_id": None,
            }
        )
    elif entry_node is None:
        errors.append(
            {
                "code": "invalid_entry_node_id",
                "message": f"entry_node_id '{entry_node_id}' does not reference an existing node",
                "node_id": None,
            }
        )
    elif entry_node.get("type") != ENTRY_NODE_TYPE:
        errors.append(
            {
                "code": "entry_node_not_greeting",
                "message": f"entry_node_id '{entry_node_id}' must reference the flow's greeting node",
                "node_id": entry_node_id,
            }
        )

    adjacency: Dict[str, List[str]] = {node_id: [] for node_id in node_by_id}
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source not in node_by_id or target not in node_by_id:
            errors.append(
                {
                    "code": "invalid_edge",
                    "message": f"Edge {edge.get('id')} references an unknown node",
                    "node_id": None,
                }
            )
            continue
        adjacency[source].append(target)

    for node_id, node in node_by_id.items():
        if node.get("type") in TERMINAL_NODE_TYPES:
            continue
        if not adjacency.get(node_id):
            errors.append(
                {
                    "code": "missing_outgoing_edge",
                    "message": f"Node {node_id} has no outgoing edges",
                    "node_id": node_id,
                }
            )

    # Cycle detection: DFS with recursion stack via 3-color marking, O(V+E).
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node_id: WHITE for node_id in node_by_id}

    def _has_cycle(node_id: str) -> bool:
        color[node_id] = GRAY
        for neighbor in adjacency.get(node_id, []):
            if color[neighbor] == GRAY:
                return True
            if color[neighbor] == WHITE and _has_cycle(neighbor):
                return True
        color[node_id] = BLACK
        return False

    for node_id in node_by_id:
        if color[node_id] == WHITE and _has_cycle(node_id):
            errors.append(
                {
                    "code": "cycle_detected",
                    "message": "Flow graph contains a cycle",
                    "node_id": None,
                }
            )
            break

    # Orphan/reachability check, only meaningful with a single, valid entry node.
    if entry_node is not None:
        visited: set = set()
        stack = [entry_node_id]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(adjacency.get(current, []))
        for node_id in node_by_id:
            if node_id not in visited:
                errors.append(
                    {
                        "code": "orphan_node",
                        "message": f"Node {node_id} is not reachable from the entry node",
                        "node_id": node_id,
                    }
                )

    return errors


def compile_graph(flow_data: Dict[str, Any]) -> Dict[str, Any]:
    """Compile raw flow_data into the pre-compiled executor lookup.

    Returns ``{node_id: {"type": str, "data": dict, "next_nodes": {handle:
    target_node_id}}}`` — a flat, call-time lookup with no re-parsing of the
    raw graph. ``next_nodes`` keys are each outgoing edge's ``sourceHandle``
    (``"yes"``/``"no"`` for ``branch`` nodes, ``"default"`` for every other
    node type when the edge doesn't specify a handle).

    Branch conditions are *not* attached to edges here — ``condition_variable``
    /``operator``/``condition_value`` live on the ``branch`` node's own
    ``data`` and are evaluated directly against it at call time (see
    ``FlowExecutor.next_node_id``), matching the ticket's per-node-type
    executor logic table.
    """
    nodes = flow_data.get("nodes") or []
    edges = flow_data.get("edges") or []

    next_nodes: Dict[str, Dict[str, str]] = {node["id"]: {} for node in nodes}
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source not in next_nodes:
            continue
        handle = edge.get("sourceHandle") or DEFAULT_HANDLE
        next_nodes[source][handle] = target

    compiled: Dict[str, Any] = {}
    for node in nodes:
        node_id = node["id"]
        compiled[node_id] = {
            "type": node.get("type", ""),
            "data": node.get("data") or {},
            "next_nodes": next_nodes.get(node_id, {}),
        }
    return compiled
