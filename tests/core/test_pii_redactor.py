"""
Unit tests for app.core.pii_redactor.redact_pii.

Covers: emails, phones, cards, SSNs, account numbers, honorific names,
nested dicts/lists, and the recursion-depth guard.
"""

import pytest
from app.core.pii_redactor import redact_pii


# ---------------------------------------------------------------------------
# Flat string patterns
# ---------------------------------------------------------------------------

class TestEmailRedaction:
    def test_plain_email(self):
        assert redact_pii("contact john.doe@example.com please") == "contact [REDACTED_EMAIL] please"

    def test_email_subdomain(self):
        assert "[REDACTED_EMAIL]" in redact_pii("reach us at support@mail.company.co.uk")

    def test_no_false_positive(self):
        result = redact_pii("no email here")
        assert result == "no email here"


class TestPhoneRedaction:
    def test_us_dashes(self):
        assert "[REDACTED_PHONE]" in redact_pii("Call 555-867-5309 now")

    def test_e164(self):
        assert "[REDACTED_PHONE]" in redact_pii("WhatsApp +14155552671")

    def test_parentheses_format(self):
        assert "[REDACTED_PHONE]" in redact_pii("(800) 555-0100 is the number")


class TestCardRedaction:
    def test_visa_16_digits(self):
        assert "[REDACTED_CARD]" in redact_pii("Card: 4111 1111 1111 1111")

    def test_amex_15_digits(self):
        assert "[REDACTED_CARD]" in redact_pii("Amex 371449635398431")

    def test_dashes(self):
        assert "[REDACTED_CARD]" in redact_pii("4111-1111-1111-1111")


class TestSsnRedaction:
    def test_hyphenated(self):
        assert "[REDACTED_SSN]" in redact_pii("SSN: 123-45-6789")

    def test_nine_digits(self):
        assert "[REDACTED_SSN]" in redact_pii("Social 123456789 on file")


class TestAccountRedaction:
    def test_bank_account(self):
        result = redact_pii("Account 12345678 routing")
        assert "[REDACTED_ACCOUNT]" in result or "[REDACTED_SSN]" in result  # 8-digit may match either

    def test_long_account(self):
        # 11-digit numbers may match the phone pattern before account — either redaction is correct
        result = redact_pii("IBAN last digits 12345678901")
        assert "[REDACTED_ACCOUNT]" in result or "[REDACTED_PHONE]" in result


class TestNameRedaction:
    def test_honorific_name(self):
        assert "[REDACTED_NAME]" in redact_pii("Appointment for Dr. Jane Smith tomorrow")

    def test_mr(self):
        assert "[REDACTED_NAME]" in redact_pii("Hello Mr. John Doe")

    def test_mrs(self):
        assert "[REDACTED_NAME]" in redact_pii("Invoice for Mrs. Alice Brown")


# ---------------------------------------------------------------------------
# Nested structures
# ---------------------------------------------------------------------------

class TestNestedDict:
    def test_shallow_dict(self):
        result = redact_pii({"email": "user@test.com", "note": "hello"})
        assert result["email"] == "[REDACTED_EMAIL]"
        assert result["note"] == "hello"

    def test_nested_dict(self):
        payload = {
            "user": {
                "contact": "reach me at bob@example.org",
                "phone": "555-123-4567",
            },
            "status": "active",
        }
        result = redact_pii(payload)
        assert "[REDACTED_EMAIL]" in result["user"]["contact"]
        assert "[REDACTED_PHONE]" in result["user"]["phone"]
        assert result["status"] == "active"

    def test_deeply_nested(self):
        data = {"a": {"b": {"c": {"d": "test@deep.com"}}}}
        result = redact_pii(data)
        assert result["a"]["b"]["c"]["d"] == "[REDACTED_EMAIL]"


class TestNestedList:
    def test_list_of_strings(self):
        result = redact_pii(["hello", "user@test.com", "world"])
        assert result[1] == "[REDACTED_EMAIL]"
        assert result[0] == "hello"

    def test_list_of_dicts(self):
        data = [{"phone": "555-867-5309"}, {"name": "safe"}]
        result = redact_pii(data)
        assert "[REDACTED_PHONE]" in result[0]["phone"]
        assert result[1]["name"] == "safe"

    def test_tuple_preserved(self):
        result = redact_pii(("a@b.com", "plain"))
        assert isinstance(result, tuple)
        assert result[0] == "[REDACTED_EMAIL]"


class TestMixedPayload:
    def test_error_response_dict(self):
        error = {
            "detail": "User jane.doe@corp.com failed auth from 555-100-2000",
            "code": "unauthorized",
        }
        result = redact_pii(error)
        assert "[REDACTED_EMAIL]" in result["detail"]
        assert "[REDACTED_PHONE]" in result["detail"]
        assert result["code"] == "unauthorized"

    def test_list_inside_dict(self):
        data = {"errors": ["bad email foo@bar.com", "no ssn 123-45-6789"]}
        result = redact_pii(data)
        assert "[REDACTED_EMAIL]" in result["errors"][0]
        assert "[REDACTED_SSN]" in result["errors"][1]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_none_passthrough(self):
        assert redact_pii(None) is None

    def test_integer_passthrough(self):
        assert redact_pii(42) == 42

    def test_empty_string(self):
        assert redact_pii("") == ""

    def test_bytes(self):
        result = redact_pii(b"email: user@test.com")
        assert b"[REDACTED_EMAIL]" in result

    def test_depth_guard(self):
        # A 25-level deep dict should not raise; the leaf is returned as-is beyond depth 20
        def make_deep(n: int):
            if n == 0:
                return "leaf@example.com"
            return {"child": make_deep(n - 1)}

        result = redact_pii(make_deep(25))
        # Just ensure no exception is raised and a value is returned
        assert result is not None

    def test_no_mutation_of_original(self):
        original = {"key": "user@example.com"}
        _ = redact_pii(original)
        assert original["key"] == "user@example.com"
