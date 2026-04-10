"""
Recover and validate email addresses from voice/STT transcripts and LLM token fields.

STT often returns spoken forms ("john at gmail dot com") or spaced spellings; we
normalize lightly, then validate with email-validator (no deliverability checks).
"""
from __future__ import annotations

import re
from typing import Optional

# Conservative: avoids matching paths, times, etc.
_EMAIL_LIKE = re.compile(
    r"\b[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9][A-Za-z0-9.-]*\.[A-Za-z]{2,}\b",
)


def _expand_spoken_forms(text: str) -> str:
    """Turn common spoken patterns into something EMAIL_LIKE can match."""
    t = (text or "").strip()
    if not t:
        return ""
    t = t.replace("’", "'").replace("`", "")
    # Order matters: replace "dot" before "at" so domain dots resolve correctly.
    t = re.sub(r"\s+dot\s+", ".", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+at\s+", "@", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", "", t)
    return t


def _validate(email: str) -> Optional[str]:
    from email_validator import EmailNotValidError, validate_email

    try:
        return validate_email(email.strip(), check_deliverability=False).normalized
    except EmailNotValidError:
        return None


def coerce_email_from_text(text: str) -> Optional[str]:
    """Return the first syntactically valid email found in *text* (plain or spoken)."""
    if not (text or "").strip():
        return None

    for m in _EMAIL_LIKE.finditer(text):
        norm = _validate(m.group(0))
        if norm:
            return norm

    expanded = _expand_spoken_forms(text)
    if expanded:
        for m in _EMAIL_LIKE.finditer(expanded):
            norm = _validate(m.group(0))
            if norm:
                return norm

    return None


def best_email_from_client_utterances(utterances_newest_first: list[str]) -> Optional[str]:
    """
    Pick the best email from recent client lines, preferring the newest valid hit.
    """
    for line in utterances_newest_first:
        hit = coerce_email_from_text(line)
        if hit:
            return hit
    return None


def resolve_customer_email_for_booking(
    *,
    token_email_raw: Optional[str],
    transcript_client_lines_newest_first: list[str],
) -> Optional[str]:
    """
    Prefer an explicit email= value from the booking token when it validates;
    otherwise scan recent client transcript lines (voice recovery).
    """
    if token_email_raw and str(token_email_raw).strip():
        t = str(token_email_raw).strip()
        lowered = t.lower()
        if lowered in ("none", "n/a", "na", "null", "-", ""):
            pass
        else:
            direct = _validate(t) or coerce_email_from_text(t)
            if direct:
                return direct

    return best_email_from_client_utterances(transcript_client_lines_newest_first)


def normalize_stored_email(raw: Optional[str]) -> Optional[str]:
    """
    Validate an email stored on a model (DB/API) before sending notifications.
    Returns normalized form or None if missing/invalid.
    """
    if not raw or not str(raw).strip():
        return None
    return _validate(str(raw).strip())
