"""
Backend-owned contact intake + booking intent on CallSession.call_metadata.

contact_intake is the primary source of truth for name/email gating.
booking_intent holds non-PII hints from BOOK_APPOINTMENT tokens (slot, reason).
"""
from __future__ import annotations

import re
import uuid
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logger import logger
from app.models.call_session import CallSession
from app.models.transcript_message import TranscriptMessage
from app.utils.spoken_email import normalize_stored_email
from app.utils.voice_contact_extraction import (
    extract_spelled_name_from_line,
    strict_contact_email_from_text,
)

CONTACT_INTAKE_KEY = "contact_intake"
BOOKING_INTENT_KEY = "booking_intent"
MAX_NAME_SPELL_FAILURES = 3

_SPELL_NAME_AGENT = re.compile(
    r"\bspell\b.*\b(name|full\s*name|first\s*name|last\s*name)\b|\b(name|full\s*name)\b.*\bspell\b",
    flags=re.IGNORECASE,
)
_SPELL_EMAIL_AGENT = re.compile(
    r"\bspell\b.*\b(e-?mail|email\s*address)\b|\b(e-?mail)\b.*\bspell\b",
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


def _extract_confirmed_name_from_agent_text(agent_text: str) -> Optional[str]:
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
    }


def _normalize_intake(raw: Optional[dict]) -> dict[str, Any]:
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
    except Exception:
        pass


def _save_booking_intent(db: Session, call_session: CallSession, intent: dict[str, Any]) -> None:
    meta = dict(call_session.call_metadata or {})
    meta[BOOKING_INTENT_KEY] = intent
    call_session.call_metadata = meta
    db.add(call_session)
    db.commit()
    try:
        db.refresh(call_session)
    except Exception:
        pass


def merge_booking_intent(
    existing: dict[str, Any],
    *,
    slot_start_iso: Optional[str] = None,
    appointment_reason: Optional[str] = None,
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
    preceding_agent_text: Optional[str],
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

    if role == "client" and text:
        prev = (preceding_agent_text or "").strip()
        awaiting = intake.get("awaiting_spell_field")

        email_context = awaiting == "email" or bool(_SPELL_EMAIL_AGENT.search(prev))
        name_context = awaiting == "name" or bool(_SPELL_NAME_AGENT.search(prev))

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

    _save_contact_intake(db, call_session, intake)


def apply_post_call_recovery(
    db: Session,
    call_session: CallSession,
    *,
    name: Optional[str] = None,
    email: Optional[str] = None,
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


def _get_preceding_agent_message(db: Session, call_session_id: uuid.UUID) -> Optional[str]:
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


def persist_booking_intent_fields(
    db: Session,
    call_session: CallSession,
    *,
    slot_start_iso: Optional[str],
    appointment_reason: Optional[str],
) -> None:
    prev = get_booking_intent(call_session)
    merged = merge_booking_intent(
        prev,
        slot_start_iso=slot_start_iso,
        appointment_reason=appointment_reason,
    )
    _save_booking_intent(db, call_session, merged)
