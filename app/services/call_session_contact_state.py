"""
Backend-owned contact intake + booking intent on CallSession.call_metadata.

contact_intake is the primary source of truth for name/email gating.
booking_intent holds non-PII hints from BOOK_APPOINTMENT tokens (slot, reason).
"""
from __future__ import annotations

import re
import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logger import logger
from app.models.call_session import CallSession
from app.models.transcript_message import TranscriptMessage
from app.utils.spoken_email import normalize_stored_email
from app.utils.voice_contact_extraction import (
    extract_spelled_name_from_line,
    is_caller_id_phone_reference,
    plausible_address_from_text,
    strict_contact_email_from_text,
    strict_contact_phone_from_text,
)

CONTACT_INTAKE_KEY = "contact_intake"
BOOKING_INTENT_KEY = "booking_intent"
# Hard retry ceiling shared by every intake field (name, phone, address).
# Kept as one constant so the four fields behave uniformly; the effective
# threshold (3) is unchanged from the original name-only MAX_NAME_SPELL_FAILURES.
MAX_FIELD_COLLECTION_FAILURES = 3

# Ceiling sentinel for phone/address: marks a field as "gave up asking" WITHOUT
# promoting whatever garbled text last failed extraction into something that
# looks like a real, usable value. build_contact_intake_prompt_block()
# special-cases this to tell the model to stop asking without repeating
# fabricated-looking data back to the caller or writing it to a CRM lead.
_CEILING_UNRESOLVED = "__unresolved__"
# Backward-compatible alias — some call sites/tests may still reference the
# original name.
MAX_NAME_SPELL_FAILURES = MAX_FIELD_COLLECTION_FAILURES

# F-04: Mid-call correction detection
_CORRECTION_SIGNALS = re.compile(
    r"\b(actually|wait|no|sorry|i\s+meant|my\s+name\s+is|it.?s\s+spelled|"
    r"that.?s\s+wrong|let\s+me\s+correct)\b",
    re.IGNORECASE,
)


def _detect_field_correction(text: str, intake: dict) -> str | None:
    """Return field name if a correction signal targets a confirmed field."""
    if not _CORRECTION_SIGNALS.search(text):
        return None
    text_lower = text.lower()
    if intake.get("name_confident") and any(
        w in text_lower for w in ["name", "called", "i am", "i'm", "it's"]
    ):
        return "name"
    if intake.get("phone_confirmed") and any(
        w in text_lower for w in ["number", "phone", "digit"]
    ):
        return "phone"
    if intake.get("email_confirmed") and "email" in text_lower:
        return "email"
    return None

_SPELL_NAME_AGENT = re.compile(
    r"\bspell\b.*\b(name|full\s*name|first\s*name|last\s*name)\b|\b(name|full\s*name)\b.*\bspell\b",
    flags=re.IGNORECASE,
)
_SPELL_EMAIL_AGENT = re.compile(
    r"\bspell\b.*\b(e-?mail|email\s*address)\b|\b(e-?mail)\b.*\bspell\b",
    flags=re.IGNORECASE,
)
# Agent asking for a phone number / address. Unlike name/email these fields
# are not typically spelled letter-by-letter, so we only need to know the
# agent *asked* — not that it asked for a spelling — to set the awaiting
# context that gates extraction on the caller's next turn.
_ASK_PHONE_AGENT = re.compile(
    r"\b(?:phone\s*number|contact\s*number|call(?:back)?\s*number|"
    r"best\s+number\s+to\s+reach\s+you|number\s+to\s+reach\s+you)\b",
    flags=re.IGNORECASE,
)
_ASK_ADDRESS_AGENT = re.compile(
    r"\b(?:service\s+address|street\s+address|mailing\s+address|"
    r"your\s+address|where\s+are\s+you\s+located)\b",
    flags=re.IGNORECASE,
)

# Vapi-style natural confirmation: agent repeats a name and caller affirms.
# Triggers we look for in the agent line (case-insensitive).
_AGENT_NAME_CONFIRM_TRIGGER = re.compile(
    r"\b(?:"
    r"your\s+name\s+is|"
    r"you\s+said\s+your\s+name\s+is|"
    r"you\s+said\s+your\s+name'?s|"
    r"to\s+confirm,?\s+your\s+name\s+is|"
    r"just\s+to\s+confirm,?\s+your\s+name\s+is|"
    r"so\s+that'?s\s+|"
    r"can\s+i\s+call\s+you|"
    r"i\s+have\s+your\s+name\s+as"
    r")\s*",
    flags=re.IGNORECASE,
)
# Caller affirmation patterns ("yes", "correct", "that's right", …).
_CLIENT_AFFIRMATION = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|correct|that'?s\s+right|that\s+is\s+right|"
    r"that'?s\s+correct|right|exactly|confirmed|absolutely|sure|"
    r"100%|hundred\s+percent)\b",
    flags=re.IGNORECASE,
)
_NAME_CANDIDATE = re.compile(
    r"([A-Z][a-zA-Z\-']{1,24}(?:\s+[A-Z][a-zA-Z\-']{1,24}){0,2})",
)
_NAME_BLOCKLIST = {
    "the", "a", "an", "is", "at", "that", "right", "correct", "confirmed",
    "ok", "okay", "yes", "no", "ai", "assistant", "agent", "bot",
}

# Caller self-introduction patterns: "My name is X", "I'm X", "I am X",
# "This is X", "Call me X", "Name's X". The captured name may be 1-2 tokens.
# We accept STT lowercase output (e.g. "my name is nishan") and re-title-case
# the candidate before storing it.
_CLIENT_SELF_INTRO_NAME = re.compile(
    r"\b(?:my\s+name(?:'?s|\s+is)|i\s+am|i'?m|this\s+is|name'?s|call\s+me)\s+"
    r"(?P<name>[A-Za-z][A-Za-z\-']{1,30}(?:\s+[A-Za-z][A-Za-z\-']{1,30})?)\b",
    flags=re.IGNORECASE,
)

# Words that frequently follow "I'm …" / "this is …" but are NOT a name.
# Conservative list — adding more here only reduces false positives.
_SELF_INTRO_NON_NAME_FIRST_WORDS = frozenset({
    # General confirmation / mood
    "the", "a", "an", "is", "at", "that", "right", "correct", "confirmed",
    "ok", "okay", "yes", "no", "ai", "assistant", "agent", "bot",
    # Mood / state words after "I'm"
    "here", "good", "fine", "great", "alright", "happy", "sad", "tired",
    "feeling", "doing", "well", "stressed", "frustrated", "angry",
    # Activities after "I'm"
    "calling", "looking", "trying", "interested", "ready", "available",
    "busy", "free", "flexible", "urgent", "needing", "wanting",
    "thinking", "wondering", "asking", "checking", "having",
    "sorry", "afraid", "stuck",
    # Context after "this is"
    "important", "regarding", "about", "for", "on", "in", "to",
    "emergency",
    # Polite trailing tokens that often appear right after a name
    # ("Call me Nishan please", "I'm Nishan thanks").
    "please", "thanks", "thank", "ty",
    # Action verbs that frequently trail a name
    "from", "with", "speaking", "calling", "here",
})


def _extract_self_intro_name(client_text: str) -> str | None:
    """
    Return a Title-Cased name candidate when the caller introduces themselves
    with phrases like "My name is X", "I'm X", "I am X", "This is X",
    "Call me X". Returns None when no plausible name follows the trigger.
    """
    text = (client_text or "").strip()
    if not text:
        return None
    m = _CLIENT_SELF_INTRO_NAME.search(text)
    if not m:
        return None
    raw = (m.group("name") or "").strip(" ,.;:-")
    if not raw or not (2 <= len(raw) <= 60):
        return None
    tokens = [t for t in raw.split() if t]
    if not tokens:
        return None
    if tokens[0].lower() in _SELF_INTRO_NON_NAME_FIRST_WORDS:
        return None
    if len(tokens) == 2 and tokens[1].lower() in _SELF_INTRO_NON_NAME_FIRST_WORDS:
        # Drop the trailing junk token: "I'm Nishan calling" -> "Nishan"
        tokens = tokens[:1]
    return " ".join(t[:1].upper() + t[1:].lower() for t in tokens)


def _agent_echoes_name(agent_text: str, candidate: str) -> bool:
    """
    True when the agent's spoken text uses the caller's self-introduced name
    as a standalone word. Used as an implicit-confirmation signal.
    """
    text = (agent_text or "").strip()
    cand = (candidate or "").strip()
    if not text or not cand:
        return False
    first = cand.split()[0]
    # Names must appear as a whole word (case-insensitive) so we don't
    # pick up substring overlaps with regular vocabulary.
    return bool(re.search(rf"\b{re.escape(first)}\b", text, flags=re.IGNORECASE))


def _extract_confirmed_name_from_agent_text(agent_text: str) -> str | None:
    """
    Pull the most recent capitalized name candidate that the agent stated
    after a confirmation trigger. Conservative: requires Title-Case tokens
    and rejects obvious non-names. Returns None if no plausible candidate.
    """
    text = (agent_text or "").strip()
    if not text:
        return None
    trigger = _AGENT_NAME_CONFIRM_TRIGGER.search(text)
    if not trigger:
        return None
    rest = text[trigger.end():].lstrip(" ,:;-")
    name_match = _NAME_CANDIDATE.match(rest)
    if not name_match:
        return None
    candidate = name_match.group(1).strip(" ,.;:-")
    if not candidate:
        return None
    tokens = [t for t in candidate.split() if t]
    if not tokens or any(tok.lower() in _NAME_BLOCKLIST for tok in tokens):
        return None
    if len(candidate) < 2 or len(candidate) > 60:
        return None
    return candidate


def default_contact_intake() -> dict[str, Any]:
    return {
        "name": None,
        "email": None,
        "name_spelled_confirmed": False,
        "email_spelled_confirmed": False,
        "name_confident": False,
        "email_validated": False,
        "email_collection": False,
        "name_spell_failures": 0,
        "awaiting_spell_field": None,
        # Caller self-introduced their name (e.g. "My name is Nishan").
        # Once True we wait for the agent to echo the name in a later turn
        # before promoting to name_confident, so STT mishears alone never
        # trigger a confident booking.
        "name_self_introduced": False,
        # Phone/address: unlike name/email these are not spelled
        # letter-by-letter, so confirmation is immediate-on-extraction
        # (format/plausibility check passes -> confirmed) rather than
        # echo-and-wait-for-no-correction. See apply_transcript_turn.
        "phone": None,
        "phone_confirmed": False,
        "phone_collection_failures": 0,
        "address": None,
        "address_confirmed": False,
        "address_collection_failures": 0,
    }


def _normalize_intake(raw: dict | None) -> dict[str, Any]:
    base = default_contact_intake()
    if isinstance(raw, dict):
        for k in base:
            if k in raw:
                base[k] = raw[k]
    return base


def get_contact_intake(call_session: CallSession) -> dict[str, Any]:
    meta = dict(call_session.call_metadata or {})
    return _normalize_intake(meta.get(CONTACT_INTAKE_KEY))


def get_booking_intent(call_session: CallSession) -> dict[str, Any]:
    meta = dict(call_session.call_metadata or {})
    raw = meta.get(BOOKING_INTENT_KEY)
    return dict(raw) if isinstance(raw, dict) else {}


def _save_contact_intake(db: Session, call_session: CallSession, intake: dict[str, Any]) -> None:
    meta = dict(call_session.call_metadata or {})
    meta[CONTACT_INTAKE_KEY] = intake
    call_session.call_metadata = meta
    db.add(call_session)
    db.commit()
    try:
        db.refresh(call_session)
    except Exception as exc:
        logger.debug("Failed to refresh call_session %s after contact intake save: %s", call_session.id, exc)


def _save_booking_intent(db: Session, call_session: CallSession, intent: dict[str, Any]) -> None:
    meta = dict(call_session.call_metadata or {})
    meta[BOOKING_INTENT_KEY] = intent
    call_session.call_metadata = meta
    db.add(call_session)
    db.commit()
    try:
        db.refresh(call_session)
    except Exception as exc:
        logger.debug("Failed to refresh call_session %s after booking intent save: %s", call_session.id, exc)


def merge_booking_intent(
    existing: dict[str, Any],
    *,
    slot_start_iso: str | None = None,
    appointment_reason: str | None = None,
) -> dict[str, Any]:
    out = dict(existing) if existing else {}
    if slot_start_iso:
        out["slot_start_iso"] = slot_start_iso
    if appointment_reason:
        out["appointment_reason"] = str(appointment_reason).strip()
    return out


def apply_transcript_turn(
    db: Session,
    call_session: CallSession,
    *,
    role: str,
    message: str,
    preceding_agent_text: str | None,
) -> None:
    """
    Update contact_intake after a transcript line is committed.
    """
    intake = get_contact_intake(call_session)
    text = (message or "").strip()

    if role == "agent" and text:
        if _SPELL_NAME_AGENT.search(text):
            intake["awaiting_spell_field"] = "name"
        elif _SPELL_EMAIL_AGENT.search(text):
            intake["awaiting_spell_field"] = "email"
        elif _ASK_PHONE_AGENT.search(text):
            intake["awaiting_spell_field"] = "phone"
        elif _ASK_ADDRESS_AGENT.search(text):
            intake["awaiting_spell_field"] = "address"

        # Implicit confirmation: caller previously self-introduced (e.g.
        # "My name is Nishan") and the agent now uses that name in its
        # response (e.g. "Okay, Nishan, what time …"). Treat this as the
        # caller's name being accepted and proceed with confident booking.
        if (
            intake.get("name_self_introduced")
            and not intake.get("name_confident")
            and intake.get("name")
            and _agent_echoes_name(text, intake.get("name") or "")
        ):
            intake["name_confident"] = True

    if role == "client" and text:
        # F-04: Mid-call correction detection — reset confirmed slot if caller corrects
        corrected_field = _detect_field_correction(text, intake)
        if corrected_field == "name":
            intake["name"] = None
            intake["name_confident"] = False
            intake["awaiting_spell_field"] = "name"
            intake["name_collection_failures"] = 0
        elif corrected_field == "phone":
            intake["phone"] = None
            intake["phone_confirmed"] = False
            intake["awaiting_spell_field"] = "phone"
            intake["phone_collection_failures"] = 0
        elif corrected_field == "email":
            intake["email"] = None
            intake["email_confirmed"] = False
            intake["awaiting_spell_field"] = "email"
            intake["email_collection_failures"] = 0

        prev = (preceding_agent_text or "").strip()
        awaiting = intake.get("awaiting_spell_field")

        email_context = awaiting == "email" or bool(_SPELL_EMAIL_AGENT.search(prev))
        name_context = awaiting == "name" or bool(_SPELL_NAME_AGENT.search(prev))
        phone_context = awaiting == "phone" or bool(_ASK_PHONE_AGENT.search(prev))
        address_context = awaiting == "address" or bool(_ASK_ADDRESS_AGENT.search(prev))

        if email_context and not name_context:
            email = strict_contact_email_from_text(text)
            if email:
                normalized_email = normalize_stored_email(email)
                if normalized_email:
                    intake["email"] = normalized_email
                    intake["email_validated"] = True
                    intake["email_spelled_confirmed"] = True
                    intake["email_collection"] = True
                else:
                    intake["email"] = email
                    intake["email_validated"] = True
                    intake["email_spelled_confirmed"] = True
                    intake["email_collection"] = True
                intake["awaiting_spell_field"] = None
            elif awaiting == "email":
                intake["awaiting_spell_field"] = None

        elif name_context:
            spelled = extract_spelled_name_from_line(text)
            if spelled and intake["name_spell_failures"] < MAX_NAME_SPELL_FAILURES:
                intake["name"] = spelled
                intake["name_spelled_confirmed"] = True
                intake["name_confident"] = True
                intake["awaiting_spell_field"] = None
            else:
                if len(text) >= 6 or len(text.split()) >= 2:
                    intake["name_spell_failures"] = min(
                        MAX_NAME_SPELL_FAILURES,
                        int(intake.get("name_spell_failures") or 0) + 1,
                    )
                if intake["name_spell_failures"] >= MAX_NAME_SPELL_FAILURES:
                    intake["name_confident"] = False
                    intake["name"] = None
                    intake["name_spelled_confirmed"] = False
                intake["awaiting_spell_field"] = None

        elif phone_context:
            # Phone is not spelled letter-by-letter, so a caller answer that
            # passes the basic format/plausibility check is confirmed
            # immediately — no echo-and-wait-for-no-correction round trip
            # (that choreography exists for name/email because STT letter
            # spelling is error-prone in a way whole-number extraction isn't).
            if is_caller_id_phone_reference(text):
                caller_number = getattr(call_session, "from_number", None) or getattr(
                    call_session, "customer_phone_number", None
                )
                if caller_number:
                    intake["phone"] = str(caller_number).strip()
                    intake["phone_confirmed"] = True
                    intake["phone_collection_failures"] = 0
                else:
                    # Reference to "the number you called me from" but no
                    # caller-id on file to resolve it against — count as a
                    # failed attempt so the agent asks the caller to just say
                    # the digits instead (mirrors the garbled "the the
                    # number we call mister" real-world case).
                    if len(text) >= 4:
                        intake["phone_collection_failures"] = min(
                            MAX_FIELD_COLLECTION_FAILURES,
                            int(intake.get("phone_collection_failures") or 0) + 1,
                        )
            else:
                phone = strict_contact_phone_from_text(text)
                if phone:
                    intake["phone"] = phone
                    intake["phone_confirmed"] = True
                    intake["phone_collection_failures"] = 0
                elif len(text) >= 4:
                    intake["phone_collection_failures"] = min(
                        MAX_FIELD_COLLECTION_FAILURES,
                        int(intake.get("phone_collection_failures") or 0) + 1,
                    )
            if (
                not intake["phone_confirmed"]
                and intake["phone_collection_failures"] >= MAX_FIELD_COLLECTION_FAILURES
            ):
                # Hard retry ceiling: stop re-asking rather than looping
                # forever on unrecognizable audio. Do NOT promote the
                # garbled/failed text as a real phone number (it already
                # failed extraction 3 times) — mark unresolved instead so
                # the prompt block tells the model to move on without
                # repeating fabricated-looking data back to the caller.
                intake["phone"] = _CEILING_UNRESOLVED
                intake["phone_confirmed"] = True
            # Only clear the awaiting-field marker once phone is settled
            # (confirmed or ceiling-accepted). Otherwise keep it set to
            # "phone" so the NEXT turn still counts as phone-collection
            # context even if the agent's retry phrasing doesn't match
            # _ASK_PHONE_AGENT (an LLM-generated re-ask can be worded many
            # ways) — this is what makes the failure ceiling reliably
            # reachable rather than resetting silently on a rephrase.
            if intake["phone_confirmed"]:
                intake["awaiting_spell_field"] = None
            else:
                intake["awaiting_spell_field"] = "phone"

        elif address_context:
            # Free-text, not letter-spelled — same immediate-confirm-on-
            # plausibility semantics as phone above, not the name/email
            # echo choreography.
            address = plausible_address_from_text(text)
            if address:
                intake["address"] = address
                intake["address_confirmed"] = True
                intake["address_collection_failures"] = 0
            elif len(text) >= 4:
                intake["address_collection_failures"] = min(
                    MAX_FIELD_COLLECTION_FAILURES,
                    int(intake.get("address_collection_failures") or 0) + 1,
                )
            if (
                not intake["address_confirmed"]
                and intake["address_collection_failures"] >= MAX_FIELD_COLLECTION_FAILURES
            ):
                # Hard retry ceiling: stop re-asking rather than looping
                # forever. Same reasoning as phone above — don't promote
                # failed/garbled text as a real address.
                intake["address"] = _CEILING_UNRESOLVED
                intake["address_confirmed"] = True
            # Same reasoning as phone above: only clear once settled, else
            # keep "address" set so the ceiling is reachable regardless of
            # how the agent rephrases its retry.
            if intake["address_confirmed"]:
                intake["awaiting_spell_field"] = None
            else:
                intake["awaiting_spell_field"] = "address"

        else:
            # Vapi-style natural confirmation: agent repeated a name and the caller
            # affirmed. Only fires when no spelling context is active and no name
            # is already confident (never overwrites stronger signals).
            if (
                getattr(settings, "VOICE_NATURAL_NAME_CONFIRMATION", True)
                and not intake.get("name_confident")
                and not intake.get("awaiting_spell_field")
                and _CLIENT_AFFIRMATION.match(text)
            ):
                candidate = _extract_confirmed_name_from_agent_text(prev)
                if candidate:
                    intake["name"] = candidate
                    intake["name_confident"] = True
                    # Deliberately do NOT set name_spelled_confirmed: this is a
                    # different (softer) provenance than letter-by-letter spelling.

        # Caller self-introduction capture ("My name is Nishan", "I'm Nishan",
        # "This is Nishan", "Call me Nishan"). Stored as the name candidate
        # plus a "name_self_introduced" flag. Confidence is upgraded later when
        # the agent echoes the name in a subsequent turn (handled in the agent
        # branch above). We never overwrite a stronger signal: if the name is
        # already confident or already spelled-confirmed, leave it alone.
        if (
            getattr(settings, "VOICE_NATURAL_NAME_CONFIRMATION", True)
            and not intake.get("name_confident")
            and not intake.get("name_spelled_confirmed")
            and not awaiting
        ):
            intro_candidate = _extract_self_intro_name(text)
            if intro_candidate:
                intake["name"] = intro_candidate
                intake["name_self_introduced"] = True

    _save_contact_intake(db, call_session, intake)


def apply_post_call_recovery(
    db: Session,
    call_session: CallSession,
    *,
    name: str | None = None,
    email: str | None = None,
    name_confident: bool = False,
    email_confident: bool = False,
) -> dict[str, Any]:
    """
    Post-call upgrade-only recovery for contact intake.

    Use AFTER the call has ended to recover signals that the strict in-call
    extractors missed (e.g. caller said "My full name is Alex Carter" and the
    agent confirmed naturally). This function NEVER downgrades existing
    confidence: it only fills in missing fields or upgrades unconfident ones.

    Returns the updated intake dict.
    """
    intake = get_contact_intake(call_session)
    changed = False
    if name and name_confident and not intake.get("name_confident"):
        clean_name = str(name).strip()
        if clean_name:
            intake["name"] = clean_name
            intake["name_confident"] = True
            changed = True
    if email and email_confident and not intake.get("email_validated"):
        clean_email = normalize_stored_email(str(email).strip())
        if clean_email:
            intake["email"] = clean_email
            intake["email_validated"] = True
            intake["email_collection"] = True
            changed = True
    if changed:
        _save_contact_intake(db, call_session, intake)
    return intake


def sync_contact_intake_after_message(
    db: Session,
    call_session_id: uuid.UUID,
    *,
    role: str,
    message: str,
) -> None:
    cs = db.query(CallSession).filter(CallSession.id == call_session_id).first()
    if not cs:
        return

    preceding_agent = (
        _get_preceding_agent_message(db, call_session_id) if role == "client" else None
    )
    apply_transcript_turn(
        db,
        cs,
        role=role,
        message=message,
        preceding_agent_text=preceding_agent,
    )


def _get_preceding_agent_message(db: Session, call_session_id: uuid.UUID) -> str | None:
    rows = (
        db.query(TranscriptMessage)
        .filter(TranscriptMessage.call_session_id == call_session_id)
        .order_by(TranscriptMessage.sequence_number.desc())
        .limit(20)
        .all()
    )
    if not rows:
        return None
    if rows[0].role != "client":
        return None
    for m in rows[1:]:
        if m.role == "agent":
            return (m.message or "").strip() or None
    return None


def merge_contact_for_post_call(
    intake: dict[str, Any],
    extracted: dict[str, Any],
) -> dict[str, Any]:
    """
    Primary: contact_intake. Fallback: deterministic extraction when intake flags allow.
    """
    name = None
    if intake.get("name_confident"):
        name = (intake.get("name") or "").strip() or None
    if not name and intake.get("name_spelled_confirmed"):
        name = (extracted.get("name") or "").strip() or None

    ex_name = extracted.get("name")
    if name and ex_name and name.lower() != str(ex_name).lower():
        logger.warning(
            "post_call contact: intake name %r != extracted %r; using intake",
            name,
            ex_name,
        )

    email = None
    if intake.get("email_validated") and intake.get("email"):
        email = str(intake["email"]).strip() or None
    elif intake.get("email_spelled_confirmed") and extracted.get("email"):
        email = str(extracted["email"]).strip() or None

    return {"customer_name": name, "customer_email": email}


def booking_allowed(intake: dict[str, Any]) -> bool:
    return bool(intake.get("name_confident")) and bool((intake.get("name") or "").strip())


def build_contact_intake_prompt_block(intake: dict[str, Any]) -> str:
    """
    Render a deterministic "already collected, do not re-ask" block from
    contact_intake state for injection into the per-turn system prompt.

    This is the structural complement to the prompt-level "NO CONFIRMATION
    LOOPS" grounding rule: instead of relying on the model to infer
    completion by re-reading raw transcript history, tell it explicitly
    which fields are already confirmed and their values. Only confirmed
    fields are listed (unconfirmed/pending fields are omitted, not called
    out as missing, so this never pressures the model to ask about fields
    unrelated to its own flow). Returns "" when nothing is confirmed yet, so
    calls that never populate contact_intake see no prompt change at all.
    """
    lines: list[str] = []
    if intake.get("name_confident") and (intake.get("name") or "").strip():
        lines.append(f"- Name: CONFIRMED ({intake['name']})")
    if intake.get("email_validated") and (intake.get("email") or "").strip():
        lines.append(f"- Email: CONFIRMED ({intake['email']})")
    if intake.get("phone_confirmed") and (intake.get("phone") or "").strip():
        if intake["phone"] == _CEILING_UNRESOLVED:
            lines.append(
                "- Phone: SKIPPED (caller unable to provide after repeated "
                "attempts — do not ask again)"
            )
        else:
            lines.append(f"- Phone: CONFIRMED ({intake['phone']})")
    if intake.get("address_confirmed") and (intake.get("address") or "").strip():
        if intake["address"] == _CEILING_UNRESOLVED:
            lines.append(
                "- Address: SKIPPED (caller unable to provide after repeated "
                "attempts — do not ask again)"
            )
        else:
            lines.append(f"- Address: CONFIRMED ({intake['address']})")
    if not lines:
        return ""
    return (
        "# ALREADY COLLECTED — DO NOT RE-ASK\n"
        "These fields are already captured for this call. Never ask for or "
        "re-confirm them again unless the caller explicitly says one is "
        "wrong; move on to whatever is still missing, or to closing the "
        "call if everything needed is here.\n" + "\n".join(lines)
    )


def persist_booking_intent_fields(
    db: Session,
    call_session: CallSession,
    *,
    slot_start_iso: str | None,
    appointment_reason: str | None,
) -> None:
    prev = get_booking_intent(call_session)
    merged = merge_booking_intent(
        prev,
        slot_start_iso=slot_start_iso,
        appointment_reason=appointment_reason,
    )
    _save_booking_intent(db, call_session, merged)
