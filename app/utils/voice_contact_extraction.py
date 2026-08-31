"""
Deterministic name/email/phone/address extraction from voice/STT client lines.

Output shape only: {"name": str | None, "email": str | None}.
Never uses LLM tokens. Prefers spelled-letter patterns and spoken-email reconstruction.
"""
from __future__ import annotations

import re
from typing import Any

from app.core.config import settings
from app.utils.spoken_email import coerce_email_from_text

# Spoken digit words -> numerals. STT usually already renders spoken digits as
# numerals ("five five five" -> "555"), but this is cheap insurance for
# providers/configs that don't, and for garbled lines like the real-world
# "Yeah. The the number we call mister I'm calling you." case where a caller
# re-states a phone number after a mis-hear.
_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "to": "2", "too": "2",
    "three": "3", "four": "4", "for": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9",
}

# Caller referring back to the number the call is already on ("the number
# you called me from", "this number", "same number", "the number I'm calling
# you from") instead of speaking digits. Callers is treated by the caller
# (call_session_contact_state) which has access to the call's own from_number.
CALLER_ID_PHONE_REFERENCE = re.compile(
    r"\b(?:"
    r"(?:the\s+)?number\s+(?:you|we)\s+(?:call(?:ed)?|are\s+calling)\s+(?:me|you|mister|us)?"
    r"|this\s+number"
    r"|same\s+number"
    r"|number\s+i'?m\s+calling\s+(?:you\s+)?from"
    r"|calling\s+(?:you\s+)?from\s+(?:this|the\s+same)\s+number"
    r")\b",
    flags=re.IGNORECASE,
)

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


def strict_contact_email_from_text(text: str) -> str | None:
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

    cleaned_candidate: str | None = None
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


def extract_spelled_name_from_line(line: str) -> str | None:
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


def _words_to_digits(text: str) -> str:
    """Replace standalone spoken-digit words with numerals, leaving everything else intact."""
    out_tokens: list[str] = []
    for tok in re.split(r"(\s+)", text or ""):
        bare = re.sub(r"[^A-Za-z]", "", tok).lower()
        out_tokens.append(_DIGIT_WORDS.get(bare, tok))
    return "".join(out_tokens)


# Digit words with no common-English-word collision — safe to treat as a
# digit anywhere. "to"/"too"/"for"/"oh" (see _DIGIT_WORDS) are homophones of
# ordinary words and must only count inside a run that already contains an
# unambiguous digit token, otherwise ordinary filler speech ("wait, no, hold
# on") can get misread as digits.
_UNAMBIGUOUS_DIGIT_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
}


def strict_contact_phone_from_text(text: str) -> str | None:
    """
    Return a normalized 10 (or 11-with-leading-1) digit US-style phone number
    from free text, or None. Handles digit runs ("555-123-4567", "5551234567")
    and spoken-digit words ("five five five one two three four five six seven").
    Does not attempt international numbers — out of scope for this heuristic.

    Only digits drawn from a single CONTIGUOUS run of digit-bearing tokens are
    accepted (a plain word, not just whitespace, breaks the run) — this stops
    self-corrections or filler speech elsewhere in the same utterance
    ("Sorry, hold on, its five five five one two three, wait no four five six
    seven") from being concatenated into a fabricated number.
    """
    raw = (text or "").strip()
    if not raw:
        return None

    words = re.findall(r"\S+", raw)
    runs: list[str] = []
    current: list[str] = []
    current_has_unambiguous = False

    def _flush() -> None:
        nonlocal current, current_has_unambiguous
        if current and current_has_unambiguous:
            runs.append("".join(current))
        current = []
        current_has_unambiguous = False

    for word in words:
        bare_alpha = re.sub(r"[^A-Za-z]", "", word).lower()
        digit_word = _DIGIT_WORDS.get(bare_alpha)
        has_digit_char = bool(re.search(r"\d", word))
        is_numeric_punct = has_digit_char and not re.search(r"[A-Za-z]", word)

        if is_numeric_punct:
            current.append(re.sub(r"\D", "", word))
            current_has_unambiguous = True
        elif digit_word is not None:
            current.append(digit_word)
            if bare_alpha in _UNAMBIGUOUS_DIGIT_WORDS:
                current_has_unambiguous = True
        else:
            _flush()
    _flush()

    for run_digits in runs:
        digits = run_digits
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        if len(digits) == 10:
            return f"{digits[0:3]}-{digits[3:6]}-{digits[6:10]}"
    return None


def is_caller_id_phone_reference(text: str) -> bool:
    """
    True when the caller refers to "the number you called me from" / "this
    number" instead of speaking digits — signal to reuse the call's own
    from_number rather than expecting a spoken number.
    """
    return bool(CALLER_ID_PHONE_REFERENCE.search((text or "").strip()))


# Common street/unit-type words. Real addresses almost always contain at
# least one of these, or lead with a number — used to reject unrelated
# free-text speech that merely happens to contain a digit and some words
# (e.g. "It has been about 3 years since the last inspection").
_ADDRESS_KEYWORDS = re.compile(
    r"\b(?:"
    r"street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|"
    r"court|ct|way|apartment|apt|building|bldg|suite|ste|floor|fl|"
    r"unit|sector|block|highway|hwy|circle|cir|place|pl|terrace"
    r")\b\.?",
    flags=re.IGNORECASE,
)


def plausible_address_from_text(text: str) -> str | None:
    """
    Free-text address heuristic — deliberately NOT a full address parser.
    Accepts a line that looks like real, substantive speech (similar spirit
    to this repo's STT final-confidence soft-fallback: "looks like real
    speech" content is good enough, no USPS-grade validation). Requires:
    at least one digit and at least one alphabetic word of reasonable
    length, a minimum overall length so short throwaway replies ("yes",
    "okay", "123") aren't mistaken for an address, AND either a recognized
    street/unit-type keyword OR the digit appearing in the first two words
    (real addresses lead with a house/apartment number) — this rejects
    unrelated speech that merely happens to contain a stray digit
    ("It has been about 3 years since the last inspection").
    """
    raw = (text or "").strip()
    if len(raw) < 8:
        return None
    if not re.search(r"\d", raw):
        return None
    if not re.search(r"[A-Za-z]{3,}", raw):
        return None

    if _ADDRESS_KEYWORDS.search(raw):
        return raw

    leading_words = re.findall(r"\S+", raw)[:2]
    if any(re.search(r"\d", w) for w in leading_words):
        return raw

    return None


def extract_contact_from_client_lines(lines_newest_first: list[str]) -> dict[str, Any]:
    """
    Scan client lines (newest first) for a strict email and a spelled name.
    """
    name: str | None = None
    email: str | None = None
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
