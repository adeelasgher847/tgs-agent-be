"""Finalize voice appointments after the call using transcript + optional in-call hold."""

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
from app.services.call_session_service import call_session_service
from app.services.openai_service import openai_service
from app.services.transcript_service import TranscriptService


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
    def _extract_from_llm(
        self,
        transcript: str,
        reserved_slot: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        if not settings.OPENAI_API_KEY:
            return {}
        system_prompt = (
            "You extract an appointment request from a phone call transcript. "
            "Return ONLY a JSON object (no markdown) with these keys: "
            '"customer_name" (string|null), '
            '"customer_phone" (string|null), '
            '"customer_email" (string|null), '
            '"appointment_reason" (string|null), '
            '"slot_start_iso" (string|null, ISO-8601 with offset or Z). '
            "Use null when unknown. "
            "The phone can include country code. "
            "If a reserved slot was provided in the user message, still fill slot_start_iso to match it when possible."
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
                max_tokens=500,
            )
            return _parse_llm_json(resp.get("content", ""))
        except Exception as exc:
            logger.warning("Post-call LLM extraction failed: %s", exc)
            return {}

    @staticmethod
    def _merge_call_metadata(
        call_session: CallSession,
        patch: Dict[str, Any],
    ) -> None:
        base: Dict[str, Any] = dict(call_session.call_metadata or {})
        base.update({k: v for k, v in patch.items() if v is not None})
        call_session.call_metadata = base  # type: ignore[assignment]

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

        meta: Dict[str, Any] = dict((res.metadata_json or {}) if res else {})
        transcript = _transcript_to_text(db, call_session_id)
        if not transcript.strip() and not meta.get("customer_name") and not res:
            self._merge_call_metadata(
                cs,
                {
                    "post_call_appointment": "skipped",
                    "post_call_appointment_detail": "no_transcript_or_intent",
                },
            )
            db.commit()
            return

        llm = self._extract_from_llm(
            transcript,
            reserved_slot=res.slot_start if res else None,
        )
        name = (llm.get("customer_name") or meta.get("customer_name") or "").strip() or None
        phone = (llm.get("customer_phone") or meta.get("customer_phone") or "").strip() or None
        email = (llm.get("customer_email") or meta.get("customer_email") or "").strip() or None
        reason = (llm.get("appointment_reason") or meta.get("appointment_reason") or "").strip() or None
        if not reason:
            reason = None
        notes = (meta.get("notes") or "").strip() or None
        if not name or not name.strip():
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

        slot_utc: Optional[datetime] = None
        if res:
            slot_utc = res.slot_start
        else:
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
