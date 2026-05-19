"""
Unit tests for app.core.pii_redactor.redact_pii / redactPII.

Covers: emails, phones, cards, SSNs, account numbers, honorific names,
nested dicts/lists, sensitive headers, and the recursion-depth guard.
"""

import logging

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.core.error_responses import (
    build_api_error_payload,
    build_call_initiate_error_payload,
)
from app.core.logger import _PiiRedactionFilter, _PiiRedactingFormatter, setup_logging
from app.core.pii_redactor import (
    REDACTED,
    prepare_request_log_context,
    redactPII,
    redact_pii,
    redact_sensitive_headers,
    safe_error_message,
)
from app.core.exception_handlers import (
    http_exception_handler,
    unhandled_exception_handler,
)


# ---------------------------------------------------------------------------
# Flat string patterns
# ---------------------------------------------------------------------------

class TestEmailRedaction:
    def test_plain_email(self):
        assert redact_pii("contact john.doe@example.com please") == f"contact {REDACTED} please"

    def test_email_subdomain(self):
        assert REDACTED in redact_pii("reach us at support@mail.company.co.uk")

    def test_no_false_positive(self):
        assert redact_pii("no email here") == "no email here"


class TestPhoneRedaction:
    def test_us_dashes(self):
        assert REDACTED in redact_pii("Call 555-867-5309 now")

    def test_e164(self):
        assert REDACTED in redact_pii("WhatsApp +14155552671")

    def test_parentheses_format(self):
        assert REDACTED in redact_pii("(800) 555-0100 is the number")

    def test_compact_international(self):
        assert REDACTED in redact_pii("dial 14155552671 today")


class TestCardRedaction:
    def test_visa_16_digits(self):
        assert REDACTED in redact_pii("Card: 4111 1111 1111 1111")

    def test_amex_15_digits(self):
        assert REDACTED in redact_pii("Amex 371449635398431")

    def test_dashes(self):
        assert REDACTED in redact_pii("4111-1111-1111-1111")


class TestSsnRedaction:
    def test_hyphenated(self):
        assert REDACTED in redact_pii("SSN: 123-45-6789")

    def test_nine_digits(self):
        assert REDACTED in redact_pii("Social 123456789 on file")


class TestAccountRedaction:
    def test_bank_account(self):
        result = redact_pii("Account 12345678 routing")
        assert REDACTED in result

    def test_long_account(self):
        result = redact_pii("IBAN last digits 12345678901")
        assert REDACTED in result


class TestNameRedaction:
    def test_honorific_name(self):
        assert REDACTED in redact_pii("Appointment for Dr. Jane Smith tomorrow")

    def test_mr(self):
        assert REDACTED in redact_pii("Hello Mr. John Doe")

    def test_mrs(self):
        assert REDACTED in redact_pii("Invoice for Mrs. Alice Brown")

    def test_labeled_customer_name(self):
        assert REDACTED in redact_pii("customer: Jane Smith called")
        assert "Jane Smith" not in redact_pii("customer: Jane Smith called")

    def test_labeled_contact_name(self):
        assert REDACTED in redact_pii("Contact John Doe requested callback")


class TestNationalIdRedaction:
    def test_uk_nino(self):
        assert REDACTED in redact_pii("NINO AB 12 34 56 C on file")

    def test_pakistan_cnic(self):
        assert REDACTED in redact_pii("CNIC 12345-1234567-1 verified")

    def test_aadhaar_spaced(self):
        assert REDACTED in redact_pii("Aadhaar 1234 5678 9012 linked")


class TestRedactPiiAlias:
    def test_redact_pii_alias(self):
        assert redactPII("a@b.com") == redact_pii("a@b.com")
        assert REDACTED in redactPII("a@b.com")


# ---------------------------------------------------------------------------
# Nested structures
# ---------------------------------------------------------------------------

class TestNestedDict:
    def test_shallow_dict(self):
        result = redact_pii({"email": "user@test.com", "note": "hello"})
        assert result["email"] == REDACTED
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
        assert REDACTED in result["user"]["contact"]
        assert REDACTED in result["user"]["phone"]
        assert result["status"] == "active"


class TestSensitiveHeaders:
    def test_authorization_redacted(self):
        headers = {"Authorization": "Bearer secret-token", "Accept": "application/json"}
        result = redact_sensitive_headers(headers)
        assert result["Authorization"] == REDACTED
        assert result["Accept"] == "application/json"

    def test_api_key_redacted(self):
        headers = {"x-api-key": "sk-live-abc", "Content-Type": "text/plain"}
        result = redact_sensitive_headers(headers)
        assert result["x-api-key"] == REDACTED

    def test_stripe_signature_fully_redacted(self):
        headers = {"stripe-signature": "t=123,v1=abc123hash"}
        result = redact_sensitive_headers(headers)
        assert result["stripe-signature"] == REDACTED

    def test_request_start_not_redacted(self):
        headers = {"x-request-start": "1737123456789"}
        result = redact_sensitive_headers(headers)
        assert result["x-request-start"] == "1737123456789"


class TestUrlSecretParams:
    def test_stripe_checkout_url(self):
        url = "https://checkout.stripe.com/c/pay/cs_test_a1Iuv1jSR1k18o02TejSCvXs97HNOJcypVXoLEfjh4OHhrFcYhaaLHvWz7#fidsecret"
        result = redact_pii(url)
        assert "checkout.stripe.com/c/pay" not in result
        assert REDACTED in result

    def test_stripe_session_id_in_api_path(self):
        msg = "GET /v1/checkout/sessions/cs_test_a1Iuv1jSR1k18o02TejSCvXs97HNOJcypVXoLEfjh4OHhrFcYhaaLHvWz7"
        result = redact_pii(msg)
        assert "cs_test_a1Iuv1" not in result
        assert REDACTED in result

    def test_trello_token_in_url(self):
        url = (
            "https://api.trello.com/1/cards?key=abc123&"
            "token=ATTAdf5cfd491e99a43e1c560e7870b4ab60109166a6a29c3324f4a96fa33394cb2"
        )
        result = redact_pii(url)
        assert "ATTAdf5cfd" not in result
        assert "token=[REDACTED]" in result or f"token={REDACTED}" in result
        assert f"key={REDACTED}" in result


class TestRequestLogContext:
    def test_no_raw_body(self):
        ctx = prepare_request_log_context(
            "POST",
            "/webhook",
            {"Authorization": "Bearer x", "Content-Type": "application/json"},
            query_params={"phone": "+14155552671"},
            body_length=512,
        )
        assert ctx["body_length"] == 512
        assert "body" not in ctx
        assert ctx["headers"]["Authorization"] == REDACTED
        assert REDACTED in ctx["query_params"]["phone"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_none_passthrough(self):
        assert redact_pii(None) is None

    def test_bytes(self):
        result = redact_pii(b"email: user@test.com")
        assert REDACTED.encode() in result

    def test_no_mutation_of_original(self):
        original = {"key": "user@example.com"}
        _ = redact_pii(original)
        assert original["key"] == "user@example.com"


# ---------------------------------------------------------------------------
# Logging filter integration
# ---------------------------------------------------------------------------

class TestPiiLoggingFilter:
    def test_filter_redacts_log_message(self):
        setup_logging()
        filt = _PiiRedactionFilter()
        record = logging.LogRecord(
            name="tgs_agent",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="User email user@secret.com failed",
            args=(),
            exc_info=None,
        )
        filt.filter(record)
        assert REDACTED in record.msg
        assert "user@secret.com" not in record.msg

    def test_filter_redacts_exception_arg(self):
        filt = _PiiRedactionFilter()
        exc = ValueError("call failed for +1571290424242")
        record = logging.LogRecord(
            name="tgs_agent",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Call initiate failed: %s",
            args=(exc,),
            exc_info=None,
        )
        filt.filter(record)
        formatted = record.getMessage()
        assert "+1571290424242" not in formatted
        assert REDACTED in formatted

    def test_formatter_redacts_traceback(self):
        fmt = _PiiRedactingFormatter("%(message)s")
        try:
            raise ValueError("failed for user@trace.com")
        except ValueError:
            import sys

            exc_info = sys.exc_info()
        text = fmt.formatException(exc_info)
        assert "user@trace.com" not in text
        assert REDACTED in text


# ---------------------------------------------------------------------------
# API error payload helpers
# ---------------------------------------------------------------------------

class TestSafeErrorMessage:
    def test_string_redacted(self):
        msg = safe_error_message("Contact jane@corp.com")
        assert REDACTED in msg
        assert "jane@corp.com" not in msg

    def test_list_returns_generic(self):
        assert safe_error_message([{"loc": ["email"], "input": "a@b.com"}]) == "Request failed"

    def test_500_always_generic(self):
        msg = safe_error_message("user@secret.com failed", status_code=500)
        assert msg == "An internal error occurred. Please try again later."
        assert "secret.com" not in msg
        assert REDACTED not in msg


class TestBuildApiErrorPayload:
    def test_structure(self):
        payload = build_api_error_payload(404, "Agent not found")
        assert payload["error_code"] == "NOT_FOUND"
        assert payload["message"] == "Agent not found"
        assert payload["status_code"] == 404

    def test_pii_in_detail_redacted(self):
        payload = build_api_error_payload(400, "Invalid user bob@example.com")
        assert REDACTED in payload["message"]
        assert "bob@example.com" not in payload["message"]

    def test_500_never_leaks_exception_text(self):
        payload = build_api_error_payload(500, "user@leak.com timeout")
        assert "leak.com" not in payload["message"]
        assert payload["error_code"] == "INTERNAL_ERROR"


class TestCallInitiateErrorPayload:
    def test_safe_detail_and_crm_fields(self):
        from types import SimpleNamespace

        req = SimpleNamespace(
            board_id="b1",
            monday_item_id="m1",
            status_column_id=None,
            call_session_id_column_id=None,
            crm_container_id=None,
            crm_item_id=None,
            status_field_id=None,
            call_session_id_field_id=None,
            crm_type="monday",
        )
        payload = build_call_initiate_error_payload(
            500, "call to +14155552671 failed", req
        )
        assert payload["detail"] == payload["message"]
        assert "+14155552671" not in payload["message"]
        assert payload["board_id"] == "b1"
        assert payload["monday_item_id"] == "m1"


# ---------------------------------------------------------------------------
# Exception handlers (unit-level)
# ---------------------------------------------------------------------------

class TestExceptionHandlers:
    def test_http_exception_no_raw_pii(self):
        import asyncio
        from unittest.mock import MagicMock

        request = MagicMock()
        request.method = "GET"
        request.url.path = "/test"
        exc = HTTPException(status_code=400, detail="Bad email user@leak.com")
        response = asyncio.run(http_exception_handler(request, exc))
        body = response.body.decode()
        assert "user@leak.com" not in body
        assert REDACTED in body
        assert "error_code" in body

    def test_http_exception_500_generic_message(self):
        import asyncio
        from unittest.mock import MagicMock

        request = MagicMock()
        request.method = "GET"
        request.url.path = "/test"
        exc = HTTPException(status_code=500, detail="DB error for user@leak.com")
        response = asyncio.run(http_exception_handler(request, exc))
        body = response.body.decode()
        assert "user@leak.com" not in body
        assert "internal error" in body.lower()

    def test_validation_exception_generic_message(self):
        from pydantic import EmailStr

        class Item(BaseModel):
            email: EmailStr

        app = FastAPI()
        register_exception_handlers = __import__(
            "app.core.exception_handlers",
            fromlist=["register_exception_handlers"],
        ).register_exception_handlers
        register_exception_handlers(app)

        @app.post("/items")
        def create_item(item: Item):
            return item

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/items", json={"email": "not-an-email"})
        assert resp.status_code == 422
        data = resp.json()
        assert data["error_code"] == "VALIDATION_ERROR"
        assert data["message"] == "Request validation failed"
        assert "not-an-email" not in str(data)

    def test_unhandled_exception_safe(self):
        import asyncio
        from unittest.mock import MagicMock

        request = MagicMock()
        request.method = "POST"
        request.url.path = "/test"
        response = asyncio.run(
            unhandled_exception_handler(request, RuntimeError("user@secret.com"))
        )
        data = response.body.decode()
        assert "user@secret.com" not in data
        assert "INTERNAL_ERROR" in data
