"""Visual Flow Editor graph validation and pre-compilation.

Pure, CPU-bound functions operating on the raw React-Flow-shaped
``{"nodes": [...], "edges": [...]}`` payload — no DB/IO. Node shape:
``{"id": str, "type": str, "data": dict}``. Edge shape:
``{"id": str, "source": str, "target": str, "condition": {"type": str, ...}}``.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Nodes that leave the flow entirely (hang up or hand off) — exempt from the
# "must have an outgoing edge" rule.
TERMINAL_NODE_TYPES = {"end_call", "transfer"}

# Lower number == evaluated first at call time (see FlowExecutor.next_node_id).
_EDGE_PRIORITY = {"intent_match": 1, "keyword": 2}
_DEFAULT_EDGE_PRIORITY = 99


def _is_start_node(node: Dict[str, Any]) -> bool:
    if node.get("type") == "start":
        return True
    data = node.get("data") or {}
    return bool(data.get("isStart") or data.get("is_start"))


def validate_graph(flow_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate a flow graph. Returns a list of error dicts (empty if valid).

    Checks: exactly one start node, no directed cycles (DFS + recursion
    stack, O(V+E)), no orphan nodes (unreachable from the start node), and
    every non-end node has at least one outgoing edge.
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

    start_nodes = [n for n in node_by_id.values() if _is_start_node(n)]
    if len(start_nodes) == 0:
        errors.append(
            {
                "code": "no_start_node",
                "message": "Flow must contain exactly one start node",
                "node_id": None,
            }
        )
    elif len(start_nodes) > 1:
        errors.append(
            {
                "code": "multiple_start_nodes",
                "message": f"Flow must contain exactly one start node, found {len(start_nodes)}",
                "node_id": None,
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

    # Orphan/reachability check, only meaningful with a single, unambiguous start node.
    if len(start_nodes) == 1:
        start_id = start_nodes[0].get("id")
        visited: set = set()
        stack = [start_id]
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
                        "message": f"Node {node_id} is not reachable from the start node",
                        "node_id": node_id,
                    }
                )

    return errors


def _edge_priority(edge: Dict[str, Any]) -> int:
    condition = edge.get("condition") or {}
    return _EDGE_PRIORITY.get(condition.get("type"), _DEFAULT_EDGE_PRIORITY)


def compile_graph(flow_data: Dict[str, Any]) -> Dict[str, Any]:
    """Compile raw flow_data into a lookup-friendly decision tree.

    Returns ``{node_id: {"node": node_dict, "outgoing_edges": [sorted_edges]}}``,
    where outgoing edges are sorted by condition specificity
    (intent_match=1, keyword=2, everything else incl. fallback=99).
    """
    nodes = flow_data.get("nodes") or []
    edges = flow_data.get("edges") or []

    outgoing: Dict[str, List[Dict[str, Any]]] = {node["id"]: [] for node in nodes}
    for edge in edges:
        source = edge.get("source")
        if source in outgoing:
            outgoing[source].append(edge)

    compiled: Dict[str, Any] = {}
    for node in nodes:
        node_id = node["id"]
        sorted_edges = sorted(outgoing.get(node_id, []), key=_edge_priority)
        compiled[node_id] = {"node": node, "outgoing_edges": sorted_edges}
    return compiled
