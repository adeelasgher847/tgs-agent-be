"""
Recover and normalize email addresses from voice/STT transcripts.

Helpers used by call flows to parse spoken addresses and (in unit tests) the
`resolve_customer_email_for_booking` scoring path.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

# Conservative: avoids matching paths, times, etc.
_EMAIL_LIKE = re.compile(
    r"\b[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9][A-Za-z0-9.-]*\.[A-Za-z]{2,}\b",
)
_SPOKEN_EMAIL_MARKERS = re.compile(
    r"\b(?:at|dot|underscore|under score|dash|hyphen|plus|gmail|yahoo|hotmail|outlook)\b",
    flags=re.IGNORECASE,
)
_EXPLICIT_CONFIRMATION_MARKERS = re.compile(
    r"\b(?:yes|yeah|yep|correct|that's right|that is right|confirmed|confirm|exactly|use that|use this|right one|right email)\b",
    flags=re.IGNORECASE,
)
_SUSPICIOUS_LOCAL_PART = re.compile(
    r"(?:[a-z]{6,}[bcdfghjklmnpqrstvwxyz]{3,}$|[bcdfghjklmnpqrstvwxyz]{5,})",
    flags=re.IGNORECASE,
)
_PLACEHOLDER_VALUES = {"none", "n/a", "na", "null", "-", ""}
_TOKEN_WORD = re.compile(r"[A-Za-z0-9_+-]+")
_LEFT_CONTEXT_STOPWORDS = {
    "a",
    "address",
    "an",
    "at",
    "call",
    "contact",
    "email",
    "id",
    "is",
    "it",
    "its",
    "mail",
    "me",
    "my",
    "on",
    "reach",
    "send",
    "that",
    "the",
    "this",
    "to",
    "use",
}
_RIGHT_CONTEXT_STOPWORDS = {
    "and",
    "bye",
    "call",
    "contact",
    "correct",
    "for",
    "is",
    "it",
    "its",
    "mail",
    "okay",
    "please",
    "thanks",
    "thank",
    "that",
    "the",
    "this",
    "to",
}
_SPOKEN_SEPARATOR_TOKENS = {"dot", "underscore", "under", "score", "dash", "hyphen", "plus"}


@dataclass(frozen=True)
class EmailObservation:
    email: str
    line: str
    line_index: int
    from_spoken_reconstruction: bool


@dataclass(frozen=True)
class BookingEmailResolution:
    verified_email: Optional[str]
    pending_email: Optional[str]
    source: str
    trust_score: int
    token_email: Optional[str]
    transcript_email: Optional[str]
    suspicious_token_email: bool
    should_attempt_llm_repair: bool
    reason: str

    @property
    def final_email(self) -> Optional[str]:
        return self.verified_email


def _expand_spoken_forms(text: str) -> str:
    """Turn common spoken patterns into something EMAIL_LIKE can match."""
    t = (text or "").strip()
    if not t:
        return ""
    t = t.replace("’", "'").replace("`", "")
    replacements = (
        (r"\s+under\s+score\s+", "_"),
        (r"\s+underscore\s+", "_"),
        (r"\s+dash\s+", "-"),
        (r"\s+hyphen\s+", "-"),
        (r"\s+plus\s+", "+"),
        # Order matters: replace "dot" before "at" so domain dots resolve correctly.
        (r"\s+dot\s+", "."),
        (r"\s+at\s+", "@"),
    )
    for pattern, replacement in replacements:
        t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", "", t)
    return t


def _validate(email: str) -> Optional[str]:
    from email_validator import EmailNotValidError, validate_email

    try:
        return validate_email(email.strip(), check_deliverability=False).normalized
    except EmailNotValidError:
        return None


def _normalize_token_placeholder(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    cleaned = str(raw).strip()
    if cleaned.lower() in _PLACEHOLDER_VALUES:
        return None
    return cleaned


def _first_literal_email(text: str) -> Optional[str]:
    if not text:
        return None
    for match in _EMAIL_LIKE.finditer(text):
        normalized = _validate(match.group(0))
        if normalized:
            return normalized
    return None


def _collect_email_observations(
    utterances_newest_first: list[str],
) -> list[EmailObservation]:
    observations: list[EmailObservation] = []
    for idx, raw_line in enumerate(utterances_newest_first):
        line = (raw_line or "").strip()
        if not line:
            continue
        literal_email = _first_literal_email(line)
        candidate = coerce_email_from_text(line)
        if not candidate:
            continue
        observations.append(
            EmailObservation(
                email=candidate,
                line=line,
                line_index=idx,
                from_spoken_reconstruction=bool(
                    _SPOKEN_EMAIL_MARKERS.search(line)
                    or literal_email is None
                    or literal_email != candidate
                ),
            )
        )
    return observations


def _spoken_email_fragments(text: str) -> list[str]:
    tokens = _TOKEN_WORD.findall(text or "")
    lower_tokens = [tok.lower() for tok in tokens]
    fragments: list[str] = []

    for idx, tok in enumerate(lower_tokens):
        if tok != "at":
            continue

        left: list[str] = []
        j = idx - 1
        while j >= 0:
            current = lower_tokens[j]
            if current in _LEFT_CONTEXT_STOPWORDS and left:
                break
            if current in _LEFT_CONTEXT_STOPWORDS:
                j -= 1
                continue
            left.append(tokens[j])
            j -= 1
        left.reverse()

        right: list[str] = []
        k = idx + 1
        dot_seen = False
        while k < len(tokens):
            current = lower_tokens[k]
            if current in _RIGHT_CONTEXT_STOPWORDS and dot_seen:
                break
            if current in _RIGHT_CONTEXT_STOPWORDS and right:
                break
            right.append(tokens[k])
            if current == "dot":
                dot_seen = True
            k += 1

        while left and lower_tokens[idx - len(left)] in _SPOKEN_SEPARATOR_TOKENS:
            left.pop(0)
        while right and right[-1].lower() in _SPOKEN_SEPARATOR_TOKENS:
            right.pop()

        if not left or not right or not dot_seen:
            continue

        fragments.append(" ".join(left + ["at"] + right))

    return fragments


def _explicitly_confirmed_email(
    utterances_newest_first: list[str],
    observations: list[EmailObservation],
) -> Optional[str]:
    if not observations:
        return None

    counts: dict[str, int] = {}
    for obs in observations:
        counts[obs.email] = counts.get(obs.email, 0) + 1
    for email, count in counts.items():
        if count >= 2:
            return email

    for obs in observations:
        if _EXPLICIT_CONFIRMATION_MARKERS.search(obs.line):
            return obs.email

    for idx, raw_line in enumerate(utterances_newest_first):
        if not _EXPLICIT_CONFIRMATION_MARKERS.search(raw_line or ""):
            continue
        for obs in observations:
            if obs.line_index > idx and (obs.line_index - idx) <= 2:
                return obs.email
    return None


def _looks_suspicious_local_part(email: str) -> bool:
    if not email or "@" not in email:
        return False
    local_part = email.split("@", 1)[0]
    if "." in local_part or "_" in local_part or "-" in local_part:
        return False
    return bool(_SUSPICIOUS_LOCAL_PART.search(local_part))


def coerce_email_from_text(text: str) -> Optional[str]:
    """Return the first syntactically valid email found in *text* (plain or spoken)."""
    if not (text or "").strip():
        return None

    literal = _first_literal_email(text)
    if literal:
        return literal

    for fragment in _spoken_email_fragments(text):
        expanded = _expand_spoken_forms(fragment)
        if not expanded:
            continue
        for m in _EMAIL_LIKE.finditer(expanded):
            norm = _validate(m.group(0))
            if norm:
                return norm

    return None


def best_email_from_client_utterances(utterances_newest_first: list[str]) -> Optional[str]:
    """
    Pick the best email from recent client lines, preferring the newest valid hit.
    """
    observations = _collect_email_observations(utterances_newest_first)
    return observations[0].email if observations else None


def resolve_customer_email_for_booking(
    *,
    token_email_raw: Optional[str],
    transcript_client_lines_newest_first: list[str],
) -> BookingEmailResolution:
    """
    Deterministic resolution for booking emails (retained for unit tests).

    Trust order:
    1. Explicitly user-confirmed transcript email
    2. Transcript-reconstructed email
    3. Raw token/STT email (low trust)
    """
    transcript_lines = [
        (line or "").strip()
        for line in transcript_client_lines_newest_first
        if (line or "").strip()
    ]
    observations = _collect_email_observations(transcript_lines)
    transcript_email = observations[0].email if observations else None
    confirmed_email = _explicitly_confirmed_email(transcript_lines, observations)

    raw_token = _normalize_token_placeholder(token_email_raw)
    token_email = coerce_email_from_text(raw_token) if raw_token else None
    token_is_literal = bool(raw_token and _EMAIL_LIKE.search(raw_token))

    suspicious_reasons: list[str] = []
    if token_email:
        if token_is_literal:
            suspicious_reasons.append("raw_token_email_is_low_trust")
        if transcript_email and token_email != transcript_email:
            suspicious_reasons.append("token_mismatch_with_transcript_reconstruction")
        if not transcript_email:
            suspicious_reasons.append("token_lacks_transcript_corroboration")
        if _looks_suspicious_local_part(token_email):
            suspicious_reasons.append("token_local_part_looks_fused_or_unusual")

    if confirmed_email:
        return BookingEmailResolution(
            verified_email=confirmed_email,
            pending_email=None,
            source="explicit_user_confirmed",
            trust_score=100,
            token_email=token_email,
            transcript_email=transcript_email,
            suspicious_token_email=bool(suspicious_reasons),
            should_attempt_llm_repair=False,
            reason="Transcript email was explicitly confirmed by the caller.",
        )

    if transcript_email:
        trust_score = 80 if observations[0].from_spoken_reconstruction else 72
        reason = "Using transcript reconstruction; verification still required before notifications."
        if suspicious_reasons:
            reason += f" Token deprioritized: {', '.join(suspicious_reasons)}."
        return BookingEmailResolution(
            verified_email=None,
            pending_email=transcript_email,
            source="transcript_reconstructed",
            trust_score=trust_score,
            token_email=token_email,
            transcript_email=transcript_email,
            suspicious_token_email=bool(suspicious_reasons),
            should_attempt_llm_repair=False,
            reason=reason,
        )

    if token_email:
        return BookingEmailResolution(
            verified_email=None,
            pending_email=None if token_is_literal else token_email,
            source="token_only_unverified",
            trust_score=25 if token_is_literal else 35,
            token_email=token_email,
            transcript_email=None,
            suspicious_token_email=True,
            should_attempt_llm_repair=True,
            reason=(
                "Token email is not trusted without transcript corroboration; "
                "attempt repair or require manual verification."
            ),
        )

    return BookingEmailResolution(
        verified_email=None,
        pending_email=None,
        source="none",
        trust_score=0,
        token_email=None,
        transcript_email=None,
        suspicious_token_email=False,
        should_attempt_llm_repair=bool(raw_token or transcript_lines),
        reason="No trustworthy email could be resolved from token or transcript.",
    )


def normalize_stored_email(raw: Optional[str]) -> Optional[str]:
    """
    Validate an email stored on a model (DB/API) before sending notifications.
    Returns normalized form or None if missing/invalid.
    """
    if not raw or not str(raw).strip():
        return None
    return _validate(str(raw).strip())
