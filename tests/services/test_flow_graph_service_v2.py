"""Unit tests for flow_graph_service: entry-node validation and compiled-plan shape."""

from __future__ import annotations

from app.services.flow_graph_service import compile_graph, validate_graph


def _flow(entry_node_id="greet", extra_nodes=None, extra_edges=None):
    nodes = [
        {"id": "greet", "type": "greeting", "data": {"message": "hi"}},
        {
            "id": "branch_node",
            "type": "branch",
            "data": {
                "condition_variable": "contract_status",
                "operator": "equals",
                "condition_value": "active",
            },
        },
        {"id": "verified", "type": "end_call", "data": {}},
        {"id": "unverified", "type": "end_call", "data": {}},
    ] + (extra_nodes or [])
    edges = [
        {"id": "e_start", "source": "greet", "target": "branch_node"},
        {
            "id": "e_yes",
            "source": "branch_node",
            "target": "verified",
            "sourceHandle": "yes",
        },
        {
            "id": "e_no",
            "source": "branch_node",
            "target": "unverified",
            "sourceHandle": "no",
        },
    ] + (extra_edges or [])
    return {"entry_node_id": entry_node_id, "nodes": nodes, "edges": edges}


class TestCompileGraphShape:
    def test_next_nodes_built_from_source_handle(self):
        compiled = compile_graph(_flow())

        assert compiled["greet"]["next_nodes"] == {"default": "branch_node"}
        assert compiled["branch_node"]["next_nodes"] == {
            "yes": "verified",
            "no": "unverified",
        }

    def test_node_type_and_data_carried_through_unmodified(self):
        compiled = compile_graph(_flow())

        assert compiled["branch_node"]["type"] == "branch"
        assert compiled["branch_node"]["data"] == {
            "condition_variable": "contract_status",
            "operator": "equals",
            "condition_value": "active",
        }

    def test_edge_missing_source_handle_defaults_to_default(self):
        data = _flow(
            extra_nodes=[{"id": "kb", "type": "kb_lookup", "data": {}}],
            extra_edges=[{"id": "e_kb", "source": "verified", "target": "kb"}],
        )
        # verified is normally terminal (end_call); give it an edge for this test only.
        compiled = compile_graph(data)

        assert compiled["verified"]["next_nodes"] == {"default": "kb"}


class TestValidateGraphEntryNode:
    def test_well_formed_flow_has_no_errors(self):
        assert validate_graph(_flow()) == []

    def test_missing_entry_node_id(self):
        data = _flow()
        del data["entry_node_id"]

        codes = [e["code"] for e in validate_graph(data)]
        assert "missing_entry_node_id" in codes

    def test_entry_node_id_references_unknown_node(self):
        data = _flow(entry_node_id="does_not_exist")

        codes = [e["code"] for e in validate_graph(data)]
        assert "invalid_entry_node_id" in codes

    def test_entry_node_id_must_reference_greeting_node(self):
        data = _flow(entry_node_id="branch_node")

        codes = [e["code"] for e in validate_graph(data)]
        assert "entry_node_not_greeting" in codes

    def test_no_greeting_node(self):
        data = _flow()
        data["nodes"][0]["type"] = "kb_lookup"

        codes = [e["code"] for e in validate_graph(data)]
        assert "no_greeting_node" in codes

    def test_multiple_greeting_nodes(self):
        data = _flow()
        data["nodes"][1]["type"] = "greeting"

        codes = [e["code"] for e in validate_graph(data)]
        assert "multiple_greeting_nodes" in codes

    def test_duplicate_source_handle_fails_validation(self):
        data = _flow(
            extra_edges=[
                {
                    "id": "e_yes_duplicate",
                    "source": "branch_node",
                    "target": "unverified",
                    "sourceHandle": "yes",
                }
            ]
        )
        errors = validate_graph(data)
        codes = [e["code"] for e in errors]
        assert "duplicate_edge_handle" in codes
        dup_err = next(e for e in errors if e["code"] == "duplicate_edge_handle")
        assert dup_err["node_id"] == "branch_node"
        assert "duplicate outgoing edges" in dup_err["message"]

    def test_stringified_json_flow_data_validates_cleanly(self):
        import json

        data_str = json.dumps(_flow())
        assert validate_graph(data_str) == []

    def test_invalid_json_string_returns_validation_error(self):
        errors = validate_graph("not a valid json {")
        assert len(errors) == 1
        assert errors[0]["code"] == "invalid_flow_data"

    def test_none_flow_data_returns_empty_graph_error(self):
        errors = validate_graph(None)
        assert len(errors) == 1
        assert errors[0]["code"] == "empty_graph"

    def test_non_dict_flow_data_returns_validation_error(self):
        errors = validate_graph(12345)  # type: ignore[arg-type]
        assert len(errors) == 1
        assert errors[0]["code"] == "invalid_flow_data"

    def test_node_as_string_does_not_crash_and_returns_invalid_node_error(self):
        # Reproduces the AttributeError: 'str' object has no attribute 'get' bug
        data = {
            "entry_node_id": "greet",
            "nodes": ["greet", "branch_node"],
            "edges": [],
        }
        errors = validate_graph(data)
        codes = [e["code"] for e in errors]
        assert "invalid_node" in codes

    def test_edge_as_string_does_not_crash_and_returns_invalid_edge_error(self):
        data = _flow()
        data["edges"].append("malformed_edge_string")
        errors = validate_graph(data)
        codes = [e["code"] for e in errors]
        assert "invalid_edge" in codes

    def test_compile_graph_with_stringified_json_and_malformed_nodes(self):
        import json

        data_str = json.dumps(_flow())
        compiled = compile_graph(data_str)
        assert "greet" in compiled

        # None or invalid input returns empty dict without crashing
        assert compile_graph(None) == {}
        assert compile_graph("invalid json") == {}
        assert compile_graph({"nodes": ["string_node"], "edges": ["string_edge"]}) == {}
