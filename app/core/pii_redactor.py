"""
Centralised PII redaction utility.

Patterns covered:
  - Email addresses
  - Phone numbers (E.164 with leading +, US NANP compact, and formatted US/intl)
  - Credit/debit card numbers (13-19 digits, optionally dash/space separated)
  - US Social Security Numbers (###-##-#### only; bare 9-digit omitted — too many false positives)
  - Bank / account numbers (8-17 consecutive digits not already matched above)
  - Full names following common honorifics (Mr / Mrs / Ms / Dr / Prof)

Phase 3 (HIPAA): extend ``_HIPAA_PATTERNS`` with clinical terms and diagnosis
codes; they are merged into the pattern list when non-empty.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

REDACTED = "[REDACTED]"

# Header names whose values must never appear in logs (secrets / session tokens).
_SENSITIVE_HEADER_KEYS: frozenset[str] = frozenset(
    {
        "authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "x-twilio-signature",
        "stripe-signature",
        "proxy-authorization",
        "x-csrf-token",
    }
)

# Header values passed through unchanged (timestamps / correlation ids — not PII).
_PASS_THROUGH_HEADER_KEYS: frozenset[str] = frozenset(
    {"x-request-start", "x-request-id"},
)

# Query-string secrets in URLs logged by httpx/urllib3 (Trello, OAuth, etc.).
_URL_SECRET_PARAM_RE = re.compile(
    r"(?i)([?&])(token|key|api_key|apikey|client_secret|access_token|auth_token|password|secret)=([^&\s\"']+)",
)

# Phase 3 (HIPAA): populate with clinical-term and diagnosis-code patterns.
_HIPAA_PATTERNS: list[tuple[str, re.Pattern[str]]] = []

# Do not treat digit runs after Stripe/Twilio-style ID prefixes as phone numbers.
_PHONE_ID_PREFIX_EXCLUSION = (
    "(?<!pi_)(?<!ch_)(?<!sub_)(?<!cus_)(?<!acct_)(?<!in_)(?<!evt_)"
    "(?<!price_)(?<!prod_)(?<!seti_)(?<!re_)(?<!pm_)(?<!src_)(?<!tok_)(?<!card_)"
    "(?<!ba_)(?<!txn_)(?<!si_)(?<!sk_)(?<!pk_)(?<!cs_)"
)
# Standalone digit run: not embedded in alphanumeric tokens or opaque IDs.
_PHONE_STANDALONE_START = "(?<![A-Za-z0-9_])"
_PHONE_STANDALONE_END = "(?![0-9])"

# ---------------------------------------------------------------------------
# Compiled patterns – ordered so more-specific patterns run first
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # E-mail
    (
        REDACTED,
        re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE),
    ),
    # Stripe Checkout hosted URLs (contain session secrets in path/query)
    (
        REDACTED,
        re.compile(r"https://checkout\.stripe\.com/[^\s\"'\\]+", re.IGNORECASE),
    ),
    # Stripe Checkout session ids (cs_test_… / cs_live_…)
    (
        REDACTED,
        re.compile(r"\bcs_(?:test|live)_[A-Za-z0-9]+\b", re.IGNORECASE),
    ),
    # Other Stripe object ids (pi_, ch_, sub_, …) — whole token before digit-based phone rules
    (
        REDACTED,
        re.compile(
            r"\b(?:pi|ch|sub|cus|acct|in|evt|price|prod|seti|re|pm|src|tok|card|ba|txn|si|sk|pk)"
            r"_[A-Za-z0-9]+\b",
            re.IGNORECASE,
        ),
    ),
    # Credit / debit card  (13-19 digits, optional separators every 4)
    (
        REDACTED,
        re.compile(r"\b(?:\d[ \-]?){13,18}\d\b"),
    ),
    # SSN: ###-##-#### only (bare 9-digit omitted — false-positives on SIDs, UUIDs, etc.)
    (
        REDACTED,
        re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b"),
    ),
    # UK National Insurance (NINO)
    (
        REDACTED,
        re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]?\b", re.IGNORECASE),
    ),
    # Pakistan CNIC #####-#######-#
    (
        REDACTED,
        re.compile(r"\b\d{5}-\d{7}-\d\b"),
    ),
    # India Aadhaar / generic 12-digit national ID (spaced or dashed)
    (
        REDACTED,
        re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
    ),
    # E.164: leading + required (avoids bare 10–15 digit IDs, timestamps, Stripe suffixes)
    (
        REDACTED,
        re.compile(
            _PHONE_ID_PREFIX_EXCLUSION
            + _PHONE_STANDALONE_START
            + r"\+[1-9]\d{9,14}"
            + _PHONE_STANDALONE_END
        ),
    ),
    # US NANP compact: 10 digits or leading 1 + 10 (no +); not 12+ digit timestamps/IDs
    (
        REDACTED,
        re.compile(
            _PHONE_ID_PREFIX_EXCLUSION
            + _PHONE_STANDALONE_START
            + r"(?:1[2-9]\d{2}[2-9]\d{6}|[2-9]\d{2}[2-9]\d{6})"
            + _PHONE_STANDALONE_END
        ),
    ),
    # Phone numbers – formatted US/intl styles
    (
        REDACTED,
        re.compile(
            r"(?:\+?\d{1,3}[\s.\-]?)?"
            r"(?:\(?\d{3}\)?[\s.\-]?)"
            r"\d{3}[\s.\-]?\d{4}"
        ),
    ),
    # Bank / account numbers: 8-17 digit sequences not caught above
    (
        REDACTED,
        re.compile(r"\b\d{8,17}\b"),
    ),
    # Labeled full names: "customer: Jane Smith", "contact John Doe"
    (
        REDACTED,
        re.compile(
            r"(?i)\b(?:name|customer|patient|contact|user|caller|client|applicant|candidate)"
            r"\s*:?\s*"
            r"[A-Z][a-z]+(?:\s+[A-Z][a-z'\-]+){0,2}",
        ),
    ),
    # Names after honorifics
    (
        REDACTED,
        re.compile(
            r"\b(?:Mr\.?|Mrs\.?|Ms\.?|Miss|Dr\.?|Prof\.?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*",
            re.IGNORECASE,
        ),
    ),
]

_GENERIC_500_MESSAGE = "An internal error occurred. Please try again later."


def _all_patterns() -> list[tuple[str, re.Pattern[str]]]:
    return _PATTERNS + _HIPAA_PATTERNS


def _redact_string(value: str) -> str:
    for replacement, pattern in _all_patterns():
        value = pattern.sub(replacement, value)
    value = _URL_SECRET_PARAM_RE.sub(rf"\1\2={REDACTED}", value)
    return value


def redact_pii(value: Any, _depth: int = 0) -> Any:
    """
    Recursively redact PII from *value*.

    Supports str, bytes, dict, list, tuple, and any other type (returned as-is).
    Recursion is capped at depth 20 to guard against pathological inputs.
    """
    if _depth > 20:
        return value

    if isinstance(value, str):
        return _redact_string(value)

    if isinstance(value, bytes):
        try:
            return _redact_string(value.decode("utf-8", errors="replace")).encode("utf-8")
        except Exception:
            return value

    if isinstance(value, dict):
        return {k: redact_pii(v, _depth + 1) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        redacted = [redact_pii(item, _depth + 1) for item in value]
        return type(value)(redacted)

    return value


# Ticket-facing alias (camelCase).
redactPII = redact_pii


def redact_sensitive_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of *headers* safe for logging (secrets fully redacted)."""
    result: dict[str, str] = {}
    for key, value in headers.items():
        lower = key.lower()
        if lower in _SENSITIVE_HEADER_KEYS:
            result[key] = REDACTED
        elif lower in _PASS_THROUGH_HEADER_KEYS:
            result[key] = value if isinstance(value, str) else str(value)
        else:
            result[key] = redact_pii(value) if isinstance(value, str) else str(value)
    return result


def prepare_request_log_context(
    method: str,
    path: str,
    headers: Mapping[str, str],
    *,
    query_params: Mapping[str, str] | None = None,
    body_length: int | None = None,
) -> dict[str, Any]:
    """
    Build a request summary dict that is safe to log.

    Never includes raw body content — only optional ``body_length``.
    """
    ctx: dict[str, Any] = {
        "method": method,
        "path": path,
        "headers": redact_sensitive_headers(headers),
    }
    if query_params is not None:
        ctx["query_params"] = redact_pii(dict(query_params))
    if body_length is not None:
        ctx["body_length"] = body_length
    return ctx


def safe_error_message(detail: Any, *, status_code: int = 400) -> str:
    """
    Convert exception detail to a single safe user-facing string.

    Complex structures are collapsed to a generic message to avoid leaking
    field-level PII from validation payloads. Server errors (5xx) always
    return a generic message — never exception text, even if redacted.
    """
    if status_code >= 500:
        return _GENERIC_500_MESSAGE
    if detail is None:
        return "Request failed"
    if isinstance(detail, str):
        return redact_pii(detail)
    if isinstance(detail, (dict, list, tuple)):
        return "Request failed"
    return redact_pii(str(detail))


_STATUS_TO_ERROR_CODE: dict[int, str] = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    422: "VALIDATION_ERROR",
    429: "TOO_MANY_REQUESTS",
    500: "INTERNAL_ERROR",
    502: "BAD_GATEWAY",
    503: "SERVICE_UNAVAILABLE",
}


def status_to_error_code(status_code: int) -> str:
    return _STATUS_TO_ERROR_CODE.get(status_code, "HTTP_ERROR")
