"""
Centralised PII redaction utility.

Patterns covered:
  - Email addresses
  - Phone numbers (E.164 and common US/intl formats)
  - Credit/debit card numbers (13-19 digits, optionally dash/space separated)
  - US Social Security Numbers (###-##-#### and plain 9-digit)
  - Bank / account numbers (8-17 consecutive digits not already matched above)
  - Full names following common honorifics (Mr / Mrs / Ms / Dr / Prof)

All public surface area is intentionally limited to ``redact_pii(value)``.
"""

import re
from typing import Any

# ---------------------------------------------------------------------------
# Compiled patterns – ordered so more-specific patterns run first
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # E-mail
    (
        "[REDACTED_EMAIL]",
        re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE),
    ),
    # Credit / debit card  (13-19 digits, optional separators every 4)
    (
        "[REDACTED_CARD]",
        re.compile(
            r"\b(?:\d[ \-]?){13,18}\d\b"
        ),
    ),
    # SSN  ###-##-####  or  9 consecutive digits
    (
        "[REDACTED_SSN]",
        re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b|\b\d{9}\b"),
    ),
    # Phone numbers – E.164, +x (xxx) xxx-xxxx, xxx-xxx-xxxx, (xxx) xxx-xxxx, etc.
    (
        "[REDACTED_PHONE]",
        re.compile(
            r"(?:\+?\d{1,3}[\s.\-]?)?"          # optional country code
            r"(?:\(?\d{3}\)?[\s.\-]?)"           # area code
            r"\d{3}[\s.\-]?\d{4}"               # 7-digit local
        ),
    ),
    # Bank / account numbers: 8-17 digit sequences not caught above
    (
        "[REDACTED_ACCOUNT]",
        re.compile(r"\b\d{8,17}\b"),
    ),
    # Names after honorifics
    (
        "[REDACTED_NAME]",
        re.compile(
            r"\b(?:Mr\.?|Mrs\.?|Ms\.?|Miss|Dr\.?|Prof\.?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*",
            re.IGNORECASE,
        ),
    ),
]


def _redact_string(value: str) -> str:
    for replacement, pattern in _PATTERNS:
        value = pattern.sub(replacement, value)
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
