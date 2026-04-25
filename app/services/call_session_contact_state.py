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

from app.core.logger import logger
from app.models.call_session import CallSession
from app.models.transcript_message import TranscriptMessage
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


def default_contact_intake() -> dict[str, Any]:
    return {
        "name": None,
        "email": None,
        "name_spelled_confirmed": False,
        "email_spelled_confirmed": False,
        "name_confident": False,
        "email_validated": False,
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
                intake["email"] = email
                intake["email_validated"] = True
                intake["email_spelled_confirmed"] = True
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

    _save_contact_intake(db, call_session, intake)


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
