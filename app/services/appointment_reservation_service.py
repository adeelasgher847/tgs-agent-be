"""In-call slot reservations (holds) until post-call appointment creation."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.core.logger import logger
from app.models.slot_reservation import SlotReservation
from app.services.calendar_service import _fmt_slot_label, calendar_service


class AppointmentReservationService:
    def get_active_for_call_session(
        self,
        db: Session,
        call_session_id: uuid.UUID,
    ) -> Optional[SlotReservation]:
        return (
            db.query(SlotReservation)
            .filter(
                SlotReservation.call_session_id == call_session_id,
                SlotReservation.status == "active",
            )
            .order_by(SlotReservation.created_at.desc())
            .first()
        )

    def release_active_for_call_session(self, db: Session, call_session_id: uuid.UUID) -> int:
        """Mark all active holds for this call as released. Returns rows updated."""
        q = (
            db.query(SlotReservation)
            .filter(
                SlotReservation.call_session_id == call_session_id,
                SlotReservation.status == "active",
            )
            .all()
        )
        n = 0
        for row in q:
            row.status = "released"
            n += 1
        if n:
            db.commit()
        return n

    def mark_consumed(self, db: Session, reservation_id: uuid.UUID) -> bool:
        row = db.query(SlotReservation).filter(SlotReservation.id == reservation_id).first()
        if not row:
            return False
        row.status = "consumed"
        db.commit()
        return True

    def upsert_active_reservation(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        call_session_id: uuid.UUID,
        agent_id: Optional[uuid.UUID],
        slot_start: datetime,
        metadata: Dict[str, Any],
    ) -> SlotReservation:
        """
        Replace any previous active hold for this call, validate the window, and insert a new active hold.
        """
        self.release_active_for_call_session(db, call_session_id)

        (
            _bh,
            _tz,
            _slot_local,
            _slot_end_local,
            slot_start_utc,
            slot_end_utc,
            _duration,
        ) = calendar_service.resolve_slot_window(
            db=db,
            tenant_id=tenant_id,
            slot_start=slot_start,
            duration_minutes=None,
        )

        calendar_service._validate_slot_bookable(
            db=db,
            tenant_id=tenant_id,
            slot_local=_slot_local,
            slot_end_local=_slot_end_local,
            slot_start_utc=slot_start_utc,
            slot_end_utc=slot_end_utc,
            bh=_bh,
            tz_info=_tz,
        )

        appt_conflict = calendar_service._get_overlapping_appointment(
            db=db,
            tenant_id=tenant_id,
            slot_start=slot_start_utc,
            slot_end=slot_end_utc,
        )
        if appt_conflict:
            raise ValueError(
                f"The {_fmt_slot_label(_slot_local)} slot is no longer available. "
                "Please choose another time."
            )

        res_conflict = calendar_service._get_overlapping_reservation(
            db=db,
            tenant_id=tenant_id,
            slot_start=slot_start_utc,
            slot_end=slot_end_utc,
        )
        if res_conflict:
            raise ValueError(
                f"The {_fmt_slot_label(_slot_local)} slot is no longer available. "
                "Please choose another time."
            )

        row = SlotReservation(
            tenant_id=tenant_id,
            call_session_id=call_session_id,
            agent_id=agent_id,
            slot_start=slot_start_utc,
            slot_end=slot_end_utc,
            status="active",
            metadata_json=metadata or {},
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        logger.info(
            "Slot reservation created: id=%s tenant=%s call_session=%s slot_start=%s",
            row.id,
            tenant_id,
            call_session_id,
            slot_start_utc,
        )
        return row


appointment_reservation_service = AppointmentReservationService()
