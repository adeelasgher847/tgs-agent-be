"""Finalize voice appointments after the call using transcript + call_metadata (backend-owned)."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logger import logger
from app.models.call_session import CallSession
from app.models.slot_reservation import SlotReservation
from app.services.appointment_reservation_service import appointment_reservation_service
from app.services.calendar_service import calendar_service
from app.services.call_session_contact_state import (
    apply_post_call_recovery,
    booking_allowed,
    get_booking_intent,
    get_contact_intake,
    merge_contact_for_post_call,
)
from app.services.call_session_service import call_session_service
from app.services.openai_service import openai_service
from app.services.transcript_service import TranscriptService
from app.utils.spoken_email import normalize_stored_email
from app.utils.voice_contact_extraction import (
    client_lines_from_transcript_text,
    extract_contact_from_client_lines,
)

_LLM_NAME_BLOCKLIST = re.compile(
    r"\b(assistant|agent|ai|bot|receptionist|customer\s+service|"
    r"the\s+caller|the\s+user)\b",
    flags=re.IGNORECASE,
)
# Cheap pre-flight: only invoke LLM recovery if the transcript could plausibly
# carry a name/email signal. Saves API cost + latency on trivial calls.
_CONTACT_HINT = re.compile(
    r"\b(name|i\s+am|my\s+name|i'?m|i\s+go\s+by|call\s+me|"
    r"email|e-?mail|@|gmail|yahoo|outlook|hotmail)\b",
    flags=re.IGNORECASE,
)


def _transcript_worth_llm_recovery(transcript: str) -> bool:
    text = (transcript or "").strip()
    if len(text) < 30:
        return False
    return bool(_CONTACT_HINT.search(text))


def _email_anchored_in_transcript(normalized_email: str, transcript: str) -> bool:
    """
    True when a validated normalized email is plausibly grounded in what the caller said,
    not purely LLM-invented. Reduces false positives vs trusting every syntactic string.
    """
    n = (normalized_email or "").strip().lower()
    if not n or "@" not in n:
        return False
    local, _, domain = n.partition("@")
    if not local or not domain:
        return False
    t_raw = (transcript or "").lower()
    t_compact = re.sub(r"[\s,;]+", "", t_raw)
    n_compact = re.sub(r"[\s,;]+", "", n)
    if n_compact in t_compact:
        return True
    t_alnum = re.sub(r"[^a-z0-9@.]+", "", t_raw)
    return local in t_alnum and domain in t_alnum


def _parse_llm_json(content: str) -> Dict[str, Any]:
    text = (content or "").strip()
    if not text:
        return {}
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
    return {}


def _transcript_to_text(db: Session, call_session_id: uuid.UUID) -> str:
    rows = TranscriptService.get_messages_by_session(db, call_session_id)
    parts: List[str] = []
    for m in rows:
        line = f"{(m.role or 'unknown').upper()}: {m.message or ''}"
        parts.append(line)
    return "\n".join(parts)


def _parse_iso_to_utc(s: Optional[str]) -> Optional[datetime]:
    if not s or not str(s).strip():
        return None
    try:
        raw = str(s).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


class PostCallAppointmentService:
    @staticmethod
    def _name_appears_in_client_lines(
        client_lines: List[str],
        candidate: str,
    ) -> bool:
        """
        True when the candidate name (first token, case-insensitive, whole word)
        appears in any client transcript line. Used as a grounding check before
        accepting a low-confidence LLM-recovered name.
        """
        cand = (candidate or "").strip()
        if not cand or not client_lines:
            return False
        first = cand.split()[0]
        if not first:
            return False
        try:
            pattern = re.compile(rf"\b{re.escape(first)}\b", re.IGNORECASE)
        except re.error:
            return False
        for line in client_lines:
            if line and pattern.search(str(line)):
                return True
        return False

    def _extract_non_pii_from_llm(
        self,
        transcript: str,
        reserved_slot: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Optional LLM pass for non-PII fields only (reason, slot hint).
        Never used as authority for customer name or email.
        """
        if not settings.OPENAI_API_KEY:
            return {}
        system_prompt = (
            "You read a phone call transcript about scheduling. "
            "Return ONLY JSON (no markdown) with keys: "
            '"appointment_reason" (string|null), '
            '"slot_start_iso" (string|null, ISO-8601 with offset or Z). '
            "Do NOT output customer_name, customer_email, or customer_phone. "
            "Use null when unknown. "
            "If a reserved slot UTC time is given in the user message, align slot_start_iso when possible."
        )
        user_body = f"Transcript:\n{transcript}\n"
        if reserved_slot is not None:
            user_body += f"\nReserved_in_call_slot_UTC: {reserved_slot.isoformat()}\n"
        try:
            resp = openai_service.chat_completion(
                messages=[{"role": "user", "content": user_body}],
                system_prompt=system_prompt,
                model_name="gpt-4o-mini",
                temperature=0.0,
                max_tokens=400,
            )
            return _parse_llm_json(resp.get("content", ""))
        except Exception as exc:
            logger.warning("Post-call LLM (non-PII) extraction failed: %s", exc)
            return {}

    @staticmethod
    def _merge_call_metadata(
        call_session: CallSession,
        patch: Dict[str, Any],
    ) -> None:
        base: Dict[str, Any] = dict(call_session.call_metadata or {})
        base.update({k: v for k, v in patch.items() if v is not None})
        call_session.call_metadata = base  # type: ignore[assignment]

    def _recover_contact_via_llm(self, transcript: str) -> Dict[str, Any]:
        """
        Strict, schema-validated LLM extraction of caller name + email used as a
        last-mile recovery when deterministic intake is not confident.

        Returns: {
            "name": str | None,
            "email": str | None,
            "name_confident": bool,
            "email_confident": bool,
        }

        Always defensive: bad/empty LLM output yields {} (no recovery applied).
        Email is re-validated locally before being marked confident.
        """
        text = (transcript or "").strip()
        if not text:
            return {}
        if not getattr(settings, "POST_CALL_LLM_CONTACT_RECOVERY", False):
            return {}
        if not settings.OPENAI_API_KEY:
            return {}
        if not _transcript_worth_llm_recovery(text):
            return {}

        system_prompt = (
            "You analyze a phone-call transcript between an AI assistant and a caller. "
            "Extract ONLY the caller's contact details (never the assistant's). "
            "Return STRICT JSON (no markdown, no prose) matching this schema:\n"
            "{\n"
            '  "name": string|null,\n'
            '  "email": string|null,\n'
            '  "name_confident": boolean,\n'
            '  "email_confident": boolean\n'
            "}\n"
            "Rules:\n"
            "- name: caller's full name as they introduced themselves. Title-cased.\n"
            "  Use null if unclear or only the assistant's name appears.\n"
            "- email: caller's email. If the speech-to-text transcript inserted commas, "
            "spaces, or stray separators inside the email, reconstruct the most likely "
            "valid form (e.g. 'ali.sa,ee,b@gmail.com' -> 'ali.saeeb@gmail.com'). "
            "Use null if no email was given or it cannot be repaired into a valid form.\n"
            "- name_confident: true only when the caller clearly stated or affirmed their "
            "own name (e.g. 'My name is X', spelled it out, or said 'yes' after the "
            "assistant repeated it).\n"
            "- email_confident: true only when the email is unambiguous and not later "
            "contradicted by the caller.\n"
            "- Never invent values. When in doubt, set the field to null and the "
            "corresponding *_confident to false."
        )
        try:
            resp = openai_service.chat_completion(
                messages=[{"role": "user", "content": f"Transcript:\n{text}"}],
                system_prompt=system_prompt,
                model_name=getattr(
                    settings,
                    "POST_CALL_LLM_CONTACT_RECOVERY_MODEL",
                    "gpt-4o-mini",
                ),
                temperature=0.0,
                max_tokens=300,
            )
            parsed = _parse_llm_json(resp.get("content", ""))
        except Exception as exc:
            logger.warning("Post-call LLM contact recovery failed: %s", exc)
            return {}

        if not isinstance(parsed, dict):
            return {}

        out: Dict[str, Any] = {
            "name": None,
            "email": None,
            "name_confident": False,
            "email_confident": False,
        }

        raw_name = parsed.get("name")
        if isinstance(raw_name, str):
            cand = raw_name.strip().strip(".,;:")
            if (
                cand
                and 2 <= len(cand) <= 80
                and not _LLM_NAME_BLOCKLIST.search(cand)
            ):
                out["name"] = cand
                out["name_confident"] = bool(parsed.get("name_confident"))

        raw_email = parsed.get("email")
        if isinstance(raw_email, str) and raw_email.strip():
            normalized = normalize_stored_email(raw_email.strip())
            if normalized:
                out["email"] = normalized
                llm_email_conf = bool(parsed.get("email_confident"))
                out["email_confident"] = llm_email_conf
                if (
                    not llm_email_conf
                    and getattr(settings, "POST_CALL_LLM_EMAIL_ANCHOR_TRUST", True)
                    and _email_anchored_in_transcript(normalized, text)
                ):
                    out["email_confident"] = True

        return out

    def process_call_session(
        self,
        db: Session,
        call_session_id: uuid.UUID,
    ) -> None:
        cs = call_session_service.get_call_session_by_id(db, call_session_id)
        if not cs:
            logger.warning("Post-call booking: no call session %s", call_session_id)
            return

        res: Optional[SlotReservation] = appointment_reservation_service.get_active_for_call_session(
            db, call_session_id
        )
        tenant_id = cs.tenant_id
        existing_appt = calendar_service.get_active_appointment_for_call_session(
            db, tenant_id, call_session_id
        )
        if existing_appt:
            if res:
                appointment_reservation_service.release_active_for_call_session(db, call_session_id)
                try:
                    db.refresh(cs)
                except Exception:
                    pass
            self._merge_call_metadata(
                cs,
                {
                    "post_call_appointment": "skipped",
                    "post_call_appointment_detail": "appointment_already_exists_for_call",
                },
            )
            db.commit()
            return

        transcript = _transcript_to_text(db, call_session_id)
        intent = get_booking_intent(cs)
        res_meta: Dict[str, Any] = dict((res.metadata_json or {}) if res else {})

        if (
            not transcript.strip()
            and not intent.get("slot_start_iso")
            and not res
            and not (cs.from_number or cs.customer_phone_number)
        ):
            self._merge_call_metadata(
                cs,
                {
                    "post_call_appointment": "skipped",
                    "post_call_appointment_detail": "no_transcript_or_intent",
                },
            )
            db.commit()
            return

        intake = get_contact_intake(cs)
        client_lines = client_lines_from_transcript_text(transcript)
        extracted = extract_contact_from_client_lines(client_lines)
        merged_contact = merge_contact_for_post_call(intake, extracted)

        # Last-mile recovery: LLM upgrades contact_intake only (never downgrades).
        # Runs when (a) name gate fails, or (b) name is already confident but email is missing
        # (POST_CALL_LLM_EMAIL_RECOVERY_WHEN_NAME_OK).
        recovered: Dict[str, Any] = {}
        run_contact_llm = False
        if (
            getattr(settings, "POST_CALL_LLM_CONTACT_RECOVERY", False)
            and (settings.OPENAI_API_KEY or "").strip()
            and _transcript_worth_llm_recovery(transcript)
        ):
            if not booking_allowed(intake):
                run_contact_llm = True
            elif getattr(settings, "POST_CALL_LLM_EMAIL_RECOVERY_WHEN_NAME_OK", True):
                has_email = bool(
                    intake.get("email_validated")
                    and (str(intake.get("email") or "")).strip()
                )
                if not has_email:
                    run_contact_llm = True

        if run_contact_llm:
            recovered = self._recover_contact_via_llm(transcript)

            # Safety net: when the LLM returned a plausible name but flagged
            # name_confident=False, accept it ANYWAY if the rest of the call
            # clearly intended a booking (slot in booking_intent OR active
            # in-call reservation) AND the candidate name actually appears in
            # the client lines. This prevents a confirmed booking from being
            # silently dropped just because the LLM hedged on confidence.
            recovered_name = (recovered.get("name") or "").strip()
            recovered_email = (recovered.get("email") or "").strip()
            llm_name_confident = bool(recovered.get("name_confident"))
            llm_email_confident = bool(recovered.get("email_confident"))

            booking_signals = bool(
                intent.get("slot_start_iso")
                or (res is not None)
                or res_meta.get("appointment_reason")
            )
            if (
                recovered_name
                and not llm_name_confident
                and booking_signals
                and self._name_appears_in_client_lines(client_lines, recovered_name)
            ):
                logger.info(
                    "Post-call contact recovery: upgrading low-confidence LLM name "
                    "%r to confident because booking_intent + transcript anchor are present "
                    "(call_session=%s)",
                    recovered_name,
                    call_session_id,
                )
                llm_name_confident = True

            if (recovered_name and llm_name_confident) or (
                recovered_email and llm_email_confident
            ):
                apply_post_call_recovery(
                    db,
                    cs,
                    name=recovered_name or None,
                    email=recovered_email or None,
                    name_confident=llm_name_confident,
                    email_confident=llm_email_confident,
                )
                try:
                    db.refresh(cs)
                except Exception:
                    pass
                intake = get_contact_intake(cs)
                merged_contact = merge_contact_for_post_call(intake, extracted)
                self._merge_call_metadata(
                    cs,
                    {
                        "post_call_contact_recovery": "llm_succeeded",
                        "post_call_contact_recovery_model": getattr(
                            settings,
                            "POST_CALL_LLM_CONTACT_RECOVERY_MODEL",
                            "gpt-4o-mini",
                        ),
                    },
                )

        if not booking_allowed(intake):
            self._merge_call_metadata(
                cs,
                {
                    "post_call_appointment": "failed",
                    "post_call_appointment_detail": "contact_not_confident",
                },
            )
            if res:
                appointment_reservation_service.release_active_for_call_session(db, call_session_id)
            db.commit()
            return

        name = (merged_contact.get("customer_name") or "").strip() or None
        if not name:
            self._merge_call_metadata(
                cs,
                {
                    "post_call_appointment": "failed",
                    "post_call_appointment_detail": "missing_customer_name",
                },
            )
            if res:
                appointment_reservation_service.release_active_for_call_session(db, call_session_id)
            db.commit()
            return

        email_raw = merged_contact.get("customer_email")
        email = (str(email_raw).strip() if email_raw else None) or None

        llm = self._extract_non_pii_from_llm(
            transcript,
            reserved_slot=res.slot_start if res else None,
        )

        phone = ((cs.from_number or "").strip() or (cs.customer_phone_number or "").strip() or None)
        if not phone or not phone.strip():
            self._merge_call_metadata(
                cs,
                {
                    "post_call_appointment": "failed",
                    "post_call_appointment_detail": "missing_customer_phone",
                },
            )
            if res:
                appointment_reservation_service.release_active_for_call_session(db, call_session_id)
            db.commit()
            return

        reason = (
            (intent.get("appointment_reason") or "").strip()
            or (res_meta.get("appointment_reason") or "").strip()
            or (llm.get("appointment_reason") or "").strip()
            or ""
        ) or None
        if not reason:
            reason = None
        notes = (res_meta.get("notes") or "").strip() or None

        slot_utc: Optional[datetime] = None
        if res:
            slot_utc = res.slot_start
        if slot_utc is None:
            slot_utc = _parse_iso_to_utc(intent.get("slot_start_iso"))
        if slot_utc is None:
            slot_utc = _parse_iso_to_utc(llm.get("slot_start_iso") if llm else None)

        if slot_utc is None:
            self._merge_call_metadata(
                cs,
                {
                    "post_call_appointment": "failed",
                    "post_call_appointment_detail": "missing_or_invalid_slot",
                },
            )
            if res:
                appointment_reservation_service.release_active_for_call_session(db, call_session_id)
            db.commit()
            return

        consuming_id: Optional[uuid.UUID] = res.id if res else None
        try:
            appt = calendar_service.book_appointment(
                db=db,
                tenant_id=tenant_id,
                customer_name=name,
                customer_phone=phone,
                slot_start=slot_utc,
                agent_id=cs.agent_id,
                call_session_id=call_session_id,
                appointment_reason=reason,
                customer_email=email,
                notes=notes,
                created_via="voice_agent",
                duration_minutes=None,
                notify_user_id=cs.user_id,
                consuming_reservation_id=consuming_id,
            )
        except ValueError as ve:
            logger.info(
                "Post-call book_appointment failed: session=%s error=%s",
                call_session_id,
                ve,
            )
            if res:
                appointment_reservation_service.release_active_for_call_session(db, call_session_id)
            self._merge_call_metadata(
                cs,
                {
                    "post_call_appointment": "failed",
                    "post_call_appointment_detail": str(ve)[:2000],
                },
            )
            db.commit()
            return
        if consuming_id:
            appointment_reservation_service.mark_consumed(db, consuming_id)
        try:
            db.refresh(cs)
        except Exception:
            pass
        self._merge_call_metadata(
            cs,
            {
                "post_call_appointment": "success",
                "post_call_appointment_id": str(appt.id),
            },
        )
        db.commit()
        logger.info("Post-call appointment created: id=%s call_session=%s", appt.id, call_session_id)


post_call_appointment_service = PostCallAppointmentService()
