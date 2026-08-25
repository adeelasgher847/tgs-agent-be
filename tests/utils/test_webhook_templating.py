"""Unit tests for app.utils.webhook_templating (System Webhooks `{{...}}`
templating). Pure functions — no DB/mocks needed.
"""

from __future__ import annotations

from app.utils.webhook_templating import render_json_template, render_template


class TestRenderTemplate:
    def test_dotted_path_lookup(self):
        context = {"_system": {"phoneNumber": "+15551234567"}}
        assert (
            render_template("Caller: {{_system.phoneNumber}}", context)
            == "Caller: +15551234567"
        )

    def test_flat_key_lookup(self):
        context = {"customer_name": "Alice"}
        assert render_template("Hello {{customer_name}}", context) == "Hello Alice"

    def test_multiple_tokens_in_one_string(self):
        context = {"first": "Jane", "last": "Doe"}
        assert render_template("{{first}} {{last}}", context) == "Jane Doe"

    def test_missing_key_renders_empty_string(self):
        context = {"customer_name": "Alice"}
        assert render_template("Hi {{missing_key}}!", context) == "Hi !"

    def test_non_dict_intermediate_renders_empty_string(self):
        context = {"_system": "not-a-dict"}
        assert render_template("{{_system.phoneNumber}}", context) == ""

    def test_missing_intermediate_dict_renders_empty_string(self):
        context = {}
        assert render_template("{{_system.phoneNumber}}", context) == ""

    def test_non_string_resolvable_value_is_coerced_via_str(self):
        context = {"call_count": 3, "is_vip": True}
        assert render_template("{{call_count}}", context) == "3"
        assert render_template("{{is_vip}}", context) == "True"

    def test_no_tokens_present_returns_unchanged(self):
        text = "This is a plain string with no tokens."
        assert render_template(text, {}) == text

    def test_none_resolved_value_renders_empty_string(self):
        context = {"agent_id": None}
        assert render_template("agent={{agent_id}}", context) == "agent="

    def test_empty_string_input_passthrough(self):
        assert render_template("", {"a": "b"}) == ""

    def test_falsy_none_input_passthrough(self):
        assert render_template(None, {"a": "b"}) is None

    def test_deeply_nested_dotted_path(self):
        context = {"_metadata": {"account": {"tier": "gold"}}}
        assert render_template("{{_metadata.account.tier}}", context) == "gold"

    def test_whitespace_inside_braces_is_tolerated(self):
        context = {"customer_name": "Bob"}
        assert render_template("{{ customer_name }}", context) == "Bob"


class TestRenderJsonTemplate:
    def test_nested_dict_and_list_structure(self):
        template = {
            "callId": "{{call_id}}",
            "nested": {"agent": "{{agent_name}}"},
            "items": ["{{item_1}}", "{{item_2}}"],
        }
        context = {
            "call_id": "abc-123",
            "agent_name": "Ava",
            "item_1": "one",
            "item_2": "two",
        }
        result = render_json_template(template, context)
        assert result == {
            "callId": "abc-123",
            "nested": {"agent": "Ava"},
            "items": ["one", "two"],
        }

    def test_only_string_leaves_are_rendered(self):
        template = {"count": 5, "enabled": True, "ratio": 1.5, "note": None}
        result = render_json_template(template, {"count": "should-not-apply"})
        assert result == {"count": 5, "enabled": True, "ratio": 1.5, "note": None}

    def test_dict_keys_are_never_rendered(self):
        template = {"{{key_token}}": "value"}
        result = render_json_template(template, {"key_token": "resolved"})
        # Keys are left untouched — only values are walked/rendered.
        assert result == {"{{key_token}}": "value"}

    def test_plain_string_template(self):
        assert render_json_template("{{x}}", {"x": "y"}) == "y"

    def test_list_of_dicts(self):
        template = [{"a": "{{a}}"}, {"b": "{{b}}"}]
        result = render_json_template(template, {"a": "1", "b": "2"})
        assert result == [{"a": "1"}, {"b": "2"}]
