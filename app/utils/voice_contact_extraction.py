"""
Deterministic name/email extraction from voice/STT client lines.

Output shape only: {"name": str | None, "email": str | None}.
Never uses LLM tokens. Prefers spelled-letter patterns and spoken-email reconstruction.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from app.core.config import settings
from app.utils.spoken_email import coerce_email_from_text

# Minimum single-letter tokens to treat a line as a spelled name
_MIN_SPELL_LETTERS = 3

# Conservative span around '@': may contain stray commas/semicolons inserted by STT.
# Anchored on a TLD (".xx{2,}") so we don't fuse unrelated tokens together.
# Whitespace is intentionally NOT in the character classes so we don't swallow
# preceding words like "My email is ...".
_SLOPPY_EMAIL_SPAN = re.compile(
    r"[A-Za-z0-9._%+\-,;]+@[A-Za-z0-9._\-,;]+\.[A-Za-z]{2,}",
)


def _clean_email_stt_artifacts(text: str) -> str:
    """
    STT often inserts stray commas/semicolons inside an email
    (e.g. "ali.sa,ee,b@gmail.com"). Strip those artifacts INSIDE the first
    email-like span only; leave the rest of the line untouched. Idempotent.
    """
    raw = text or ""
    if not raw or "@" not in raw:
        return raw
    match = _SLOPPY_EMAIL_SPAN.search(raw)
    if not match:
        return raw
    sloppy = match.group(0)
    cleaned = re.sub(r"[,;]+", "", sloppy)
    if not cleaned or cleaned == sloppy:
        return raw
    if cleaned.count("@") != 1:
        return raw
    return raw[: match.start()] + cleaned + raw[match.end():]


def strict_contact_email_from_text(text: str) -> Optional[str]:
    """
    Return normalized email or None. Rules: exactly one '@', at least one '.',
    syntactically valid via email_validator (via spoken_email helpers).

    When EMAIL_STT_CLEANUP_ENABLED is on (default), an STT-artifact cleanup pass
    runs in parallel and is preferred when it yields a strictly longer / more
    specific email than the raw match. This recovers cases like
    "ali.sa,ee,b@gmail.com" where the literal regex would otherwise lock onto
    just "b@gmail.com" (the substring after the last comma).
    """
    if not (text or "").strip():
        return None
    candidate = coerce_email_from_text(text)

    cleaned_candidate: Optional[str] = None
    if getattr(settings, "EMAIL_STT_CLEANUP_ENABLED", True):
        cleaned_text = _clean_email_stt_artifacts(text)
        if cleaned_text != text:
            cleaned_candidate = coerce_email_from_text(cleaned_text)

    chosen = candidate
    if cleaned_candidate and (
        not candidate or len(cleaned_candidate) > len(candidate)
    ):
        chosen = cleaned_candidate

    if not chosen:
        return None
    if chosen.count("@") != 1:
        return None
    local, _, domain = chosen.partition("@")
    if not local or not domain or "." not in domain:
        return None
    return chosen


def extract_spelled_name_from_line(line: str) -> Optional[str]:
    """
    If the line looks like letter-by-letter spelling (e.g. "J O H N"),
    join into a single capitalized word. Returns None if the pattern is weak.
    """
    raw = (line or "").strip()
    if not raw:
        return None

    words = re.split(r"[\s,;]+", raw)
    letters: list[str] = []
    single_letter_words = 0
    noise = {
        "a",
        "i",
        "the",
        "is",
        "it",
        "as",
        "at",
        "an",
        "am",
        "ok",
        "yes",
        "no",
        "uh",
        "um",
        "and",
        "or",
        "my",
        "name",
        "its",
        "it's",
        "im",
        "i'm",
    }

    for w in words:
        w_clean = re.sub(r"[^A-Za-z]", "", w)
        if not w_clean:
            continue
        low = w_clean.lower()
        if low in noise:
            continue
        if len(w_clean) == 1:
            letters.append(w_clean.upper())
            single_letter_words += 1
        else:
            # Long tokens break strict spelling run (e.g. "John" mid spelling)
            if len(letters) >= _MIN_SPELL_LETTERS:
                break
            letters = []
            single_letter_words = 0

    if len(letters) < _MIN_SPELL_LETTERS:
        return None
    if single_letter_words < _MIN_SPELL_LETTERS:
        return None

    assembled = "".join(letters)
    if len(assembled) < _MIN_SPELL_LETTERS:
        return None
    return assembled[:1].upper() + assembled[1:].lower()


def extract_contact_from_client_lines(lines_newest_first: list[str]) -> dict[str, Any]:
    """
    Scan client lines (newest first) for a strict email and a spelled name.
    """
    name: Optional[str] = None
    email: Optional[str] = None
    for line in lines_newest_first:
        if not line or not str(line).strip():
            continue
        if email is None:
            email = strict_contact_email_from_text(line)
        if name is None:
            name = extract_spelled_name_from_line(line)
        if name and email:
            break
    return {"name": name, "email": email}


def client_lines_from_transcript_text(transcript_text: str) -> list[str]:
    """
    Parse CLIENT: lines from post-call transcript blob (newest block first for extraction).
    """
    lines: list[str] = []
    for block in (transcript_text or "").splitlines():
        b = (block or "").strip()
        if b.upper().startswith("CLIENT:"):
            lines.append(b.split(":", 1)[1].strip())
    # Newest-first: last line in file is most recent
    return list(reversed(lines))
