"""
Calendar Service
Local read/log CRUD for the `appointment` table only.

Availability computation, business hours, blocked slots, and in-call slot
reservations have moved to Calendly (see app/services/calendly_service.py).
This service no longer validates slot windows against local business hours —
Calendly owns conflict checking, timezone handling, and calendar sync.
All operations remain scoped to tenant_id for multi-tenant isolation.
"""
import html
import uuid
from datetime import datetime, date, timedelta, timezone, time as dt_time, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import List, Optional, Tuple

from jose import JWTError, jwt
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.core.config import settings
from app.core.logger import logger
from app.models.appointment import Appointment
from app.models.business_hours import BusinessHours
from app.models.call_session import CallSession
from app.models.user import User
from app.schemas.calendar import AppointmentOut, AvailableSlot, AvailableSlotsResponse
from app.services.business_hours_service import business_hours_service
from app.services.email_service import email_service
from app.utils.spoken_email import normalize_stored_email

DEFAULT_APPOINTMENT_DURATION_MINUTES = 30
APPOINTMENT_REVIEW_TOKEN_TTL_HOURS = 24 * 7
SLOT_BOOKING_BUFFER_MINUTES = 15


def _safe_tz(tz_str: str) -> tzinfo:
    try:
        return ZoneInfo(tz_str)
    except (ZoneInfoNotFoundError, Exception):
        return timezone.utc


ALLOWED_STATUS_TRANSITIONS = {
    "pending":   {"confirmed", "cancelled"},
    "confirmed": {"completed", "cancelled", "no_show"},
    "cancelled": set(),
    "completed": set(),
    "no_show":   set(),
}


def _fmt_slot_label(dt: datetime) -> str:
    """Format datetime to '9:00 AM' (no leading zero)."""
    return dt.strftime("%I:%M %p").lstrip("0") or "12:00 AM"


def _ensure_utc(dt_val: datetime) -> datetime:
    """Normalise a datetime to UTC. Treats naive datetimes as UTC."""
    if dt_val.tzinfo is None:
        return dt_val.replace(tzinfo=timezone.utc)
    return dt_val.astimezone(timezone.utc)


class CalendarService:

    # ── Internal helpers ────────────────────────────────────────────────────

    def _resolve_notification_email(
        self,
        db: Session,
        notify_user_id: Optional[uuid.UUID] = None,
        call_session_id: Optional[uuid.UUID] = None,
    ) -> Optional[str]:
        if notify_user_id:
            user = db.query(User).filter(User.id == notify_user_id).first()
            if user and user.email:
                return user.email
        if call_session_id:
            cs = db.query(CallSession).filter(CallSession.id == call_session_id).first()
            if cs and cs.user_id:
                user = db.query(User).filter(User.id == cs.user_id).first()
                if user and user.email:
                    return user.email
        return None

    def _send_appointment_confirmation_email(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        appt: Appointment,
        notify_user_id: Optional[uuid.UUID] = None,
        call_session_id: Optional[uuid.UUID] = None,
    ) -> None:
        """
        Send tenant-facing review request email for pending appointments.
        Customer confirmation is intentionally deferred until explicit confirmation.
        """
        if appt.status != "pending":
            return

        staff_recipient = self._resolve_notification_email(
            db=db,
            notify_user_id=notify_user_id,
            call_session_id=call_session_id,
        )
        if not staff_recipient:
            return

        tz_label, start_local, end_local = self.appointment_local_display(db, tenant_id, appt)
        time_line = (
            f"{start_local.strftime('%A, %B %d, %Y %I:%M %p')} – "
            f"{end_local.strftime('%I:%M %p')} ({tz_label})"
        )
        safe_name = html.escape(appt.customer_name or "")
        safe_reason = html.escape((appt.appointment_reason or "").strip() or "N/A")

        review_token = self._create_appointment_review_token(
            appointment_id=appt.id,
            tenant_id=tenant_id,
            reviewer_user_id=notify_user_id,
        )
        backend_base = (settings.WEBHOOK_BASE_URL or "").rstrip("/")
        review_link = (
            f"{backend_base}/api/v1/calendar/appointments/acknowledge?token={review_token}"
            if backend_base
            else ""
        )

        subject = "Assistly | Appointment pending — confirmation required"
        action_html = (
            f'<p><a href="{html.escape(review_link)}" '
            'style="display:inline-block;padding:10px 16px;background:#2563eb;color:#fff;'
            'text-decoration:none;border-radius:6px;">View & Acknowledge</a></p>'
            if review_link
            else "<p><em>Review link unavailable: WEBHOOK_BASE_URL is not configured.</em></p>"
        )

        staff_body = f"""
            <html>
            <body>
                <h2>Appointment pending — action required</h2>
                <p>Customer: <strong>{safe_name}</strong></p>
                <p>Reason: <strong>{safe_reason}</strong></p>
                <p>Time: <strong>{html.escape(time_line)}</strong></p>
                {action_html}
                <p>Opening the link will confirm this appointment.</p>
                <p>Thank you,<br>Assistly</p>
            </body>
            </html>
            """
        try:
            email_service.send_generic_email(
                to_email=staff_recipient,
                subject=subject,
                html_body=staff_body,
            )
        except Exception:
            logger.exception(
                "Staff review-request email failed for appointment=%s",
                appt.id,
            )

    def _create_appointment_review_token(
        self,
        *,
        appointment_id: uuid.UUID,
        tenant_id: uuid.UUID,
        reviewer_user_id: Optional[uuid.UUID] = None,
    ) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            "type": "appointment_review_ack",
            "appointment_id": str(appointment_id),
            "tenant_id": str(tenant_id),
            "reviewer_user_id": str(reviewer_user_id) if reviewer_user_id else None,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(hours=APPOINTMENT_REVIEW_TOKEN_TTL_HOURS)).timestamp()),
        }
        return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

    def _decode_appointment_review_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        except JWTError as e:
            raise ValueError("Invalid or expired review token.") from e
        if payload.get("type") != "appointment_review_ack":
            raise ValueError("Invalid review token type.")
        return payload

    def _send_customer_review_ack_email(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        appt: Appointment,
    ) -> None:
        customer_recipient = normalize_stored_email(appt.customer_email)
        if not customer_recipient:
            return
        tz_label, start_local, end_local = self.appointment_local_display(db, tenant_id, appt)
        time_line = (
            f"{start_local.strftime('%A, %B %d, %Y %I:%M %p')} – "
            f"{end_local.strftime('%I:%M %p')} ({tz_label})"
        )
        safe_name = html.escape(appt.customer_name or "")
        safe_reason = html.escape((appt.appointment_reason or "").strip() or "N/A")
        body = f"""
            <html>
            <body>
                <h2>Your appointment is confirmed</h2>
                <p>Hi {safe_name},</p>
                <p>Your appointment has been confirmed by our team:</p>
                <p><strong>{html.escape(time_line)}</strong></p>
                <p>Reason: <strong>{safe_reason}</strong></p>
                <p>Thank you,<br>Assistly</p>
            </body>
            </html>
            """
        email_service.send_generic_email(
            to_email=customer_recipient,
            subject="Assistly | Your appointment is confirmed",
            html_body=body,
        )

    def _notify_customer_confirmation_if_needed(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        appt: Appointment,
    ) -> bool:
        should_notify_customer = (
            normalize_stored_email(appt.customer_email) is not None
            and appt.customer_notified_on_review_at is None
        )
        if not should_notify_customer:
            return False
        try:
            self._send_customer_review_ack_email(db=db, tenant_id=tenant_id, appt=appt)
            appt.customer_notified_on_review_at = datetime.now(timezone.utc)
            return True
        except Exception:
            logger.exception(
                "Customer confirmation email failed for appointment=%s",
                appt.id,
            )
            return False

    def acknowledge_appointment_from_token(
        self,
        *,
        db: Session,
        token: str,
    ) -> Appointment:
        payload = self._decode_appointment_review_token(token)
        try:
            appt_id = uuid.UUID(payload["appointment_id"])
            tenant_id = uuid.UUID(payload["tenant_id"])
            reviewer_raw = payload.get("reviewer_user_id")
            reviewer_user_id = uuid.UUID(reviewer_raw) if reviewer_raw else None
        except Exception as e:
            raise ValueError("Malformed review token payload.") from e

        appt = self.get_appointment_by_id(db, appt_id, tenant_id)
        if not appt:
            raise ValueError("Appointment not found for review acknowledgement.")

        was_confirmed_before_ack = appt.status == "confirmed"
        changed = False
        if appt.status == "cancelled":
            raise ValueError("Cancelled appointment cannot be confirmed from acknowledgement link.")

        if appt.status != "confirmed":
            appt.status = "confirmed"
            changed = True

        if appt.reviewed_at is None:
            appt.reviewed_at = datetime.now(timezone.utc)
            changed = True
        if reviewer_user_id and appt.reviewed_by_user_id is None:
            appt.reviewed_by_user_id = reviewer_user_id
            changed = True

        if self._notify_customer_confirmation_if_needed(db=db, tenant_id=tenant_id, appt=appt):
            changed = True

        if changed:
            db.commit()
            db.refresh(appt)

        if (not was_confirmed_before_ack) and appt.status == "confirmed":
            from app.services.appointment_follow_up_service import schedule_follow_up_after_confirm

            acting = reviewer_user_id or appt.reviewed_by_user_id
            schedule_follow_up_after_confirm(db, appt, acting)

        return appt

    # ── Legacy slot availability (non-Calendly tenants only) ───────────────────
    # Calendly-enabled tenants use calendly_service.get_available_slots instead.
    # This path no longer excludes BlockedSlot ranges or in-call SlotReservation
    # holds (both removed) — only BusinessHours windows and already-booked
    # Appointment rows are honored.

    def get_tenant_timezone(self, db: Session, tenant_id: uuid.UUID) -> str:
        return business_hours_service.get_tenant_timezone(db, tenant_id)

    def _get_business_hours_for_date(
        self, db: Session, tenant_id: uuid.UUID, target_date: date
    ) -> Optional[BusinessHours]:
        return (
            db.query(BusinessHours)
            .filter(
                BusinessHours.tenant_id == tenant_id,
                BusinessHours.day_of_week == target_date.weekday(),
                BusinessHours.is_deleted.is_(False),
            )
            .first()
        )

    def get_available_slots(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        target_date: date,
        agent_id: Optional[uuid.UUID] = None,
    ) -> AvailableSlotsResponse:
        """
        Legacy fallback for tenants that have not enabled Calendly. Returns
        bookable slots for a given date, excluding past slots (with buffer)
        and already-booked appointments.
        """
        _ = agent_id

        bh = self._get_business_hours_for_date(db, tenant_id, target_date)

        empty = AvailableSlotsResponse(date=target_date.isoformat(), timezone="UTC", slots=[], total=0)
        if not bh or bh.is_closed or not bh.open_time or not bh.close_time:
            return empty

        tz_info = _safe_tz(bh.timezone)
        duration = timedelta(minutes=bh.slot_duration_minutes)
        now_utc = datetime.now(timezone.utc)
        buffer = timedelta(minutes=SLOT_BOOKING_BUFFER_MINUTES)

        cursor = datetime.combine(target_date, bh.open_time, tzinfo=tz_info)
        boundary = datetime.combine(target_date, bh.close_time, tzinfo=tz_info)
        all_slots: List[tuple] = []
        while cursor + duration <= boundary:
            all_slots.append((cursor, cursor + duration))
            cursor += duration

        if not all_slots:
            return AvailableSlotsResponse(date=target_date.isoformat(), timezone=bh.timezone, slots=[], total=0)

        day_start = datetime.combine(target_date, dt_time.min, tzinfo=tz_info)
        day_end = datetime.combine(target_date, dt_time.max, tzinfo=tz_info)

        booked = (
            db.query(Appointment.slot_start, Appointment.slot_end)
            .filter(
                Appointment.tenant_id == tenant_id,
                Appointment.slot_start < day_end,
                Appointment.slot_end > day_start,
                Appointment.status.notin_(["cancelled"]),
            )
            .all()
        )
        booked_ranges = [(_ensure_utc(bs), _ensure_utc(be)) for bs, be in booked]

        available: List[AvailableSlot] = []
        for s_start, s_end in all_slots:
            s_utc = s_start.astimezone(timezone.utc)
            s_end_utc = s_end.astimezone(timezone.utc)

            if s_utc <= now_utc + buffer:
                continue

            if any(bs < s_end_utc and be > s_utc for bs, be in booked_ranges):
                continue

            available.append(AvailableSlot(
                slot_start=s_start,
                slot_end=s_end,
                slot_label=_fmt_slot_label(s_start),
            ))

        return AvailableSlotsResponse(
            date=target_date.isoformat(),
            timezone=bh.timezone,
            slots=available,
            total=len(available),
        )

    # ── Local read-log CRUD (appointments only — availability lives in Calendly) ──
    #
    # For Calendly-connected tenants, Calendly is the source of truth for slot
    # conflicts and business hours. But this service is also the *only* write
    # path for two flows Calendly never sees: the local web booking endpoint
    # (POST /api/v1/calendar/appointments) and the legacy voice path for
    # tenants that haven't enabled Calendly — so it still guards against
    # past/too-soon slots and overlapping appointments itself.

    def _get_overlapping_appointment(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        slot_start: datetime,
        slot_end: datetime,
        exclude_appointment_id: Optional[uuid.UUID] = None,
    ) -> Optional[Appointment]:
        q = db.query(Appointment).filter(
            Appointment.tenant_id == tenant_id,
            Appointment.status.notin_(["cancelled"]),
            Appointment.slot_start < slot_end,
            Appointment.slot_end > slot_start,
        )
        if exclude_appointment_id is not None:
            q = q.filter(Appointment.id != exclude_appointment_id)
        return q.order_by(Appointment.slot_start.asc()).first()

    def _validate_local_slot_bookable(
        self,
        slot_start_utc: datetime,
        slot_end_utc: datetime,
        db: Session,
        tenant_id: uuid.UUID,
        exclude_appointment_id: Optional[uuid.UUID] = None,
    ) -> None:
        """Raises ValueError if the slot is in the past or overlaps another appointment."""
        now_utc = datetime.now(timezone.utc)
        buffer = timedelta(minutes=SLOT_BOOKING_BUFFER_MINUTES)
        if slot_start_utc <= now_utc + buffer:
            raise ValueError(
                "Cannot book a slot in the past or within the next "
                f"{SLOT_BOOKING_BUFFER_MINUTES} minutes. Please choose a later time."
            )
        conflict = self._get_overlapping_appointment(
            db, tenant_id, slot_start_utc, slot_end_utc,
            exclude_appointment_id=exclude_appointment_id,
        )
        if conflict:
            raise ValueError(
                f"The {_fmt_slot_label(slot_start_utc)} slot is no longer available. "
                "Please choose another time."
            )

    def book_appointment(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        customer_name: str,
        customer_phone: str,
        slot_start: datetime,
        agent_id: Optional[uuid.UUID] = None,
        call_session_id: Optional[uuid.UUID] = None,
        appointment_reason: Optional[str] = None,
        customer_email: Optional[str] = None,
        notes: Optional[str] = None,
        created_via: str = "voice_agent",
        duration_minutes: Optional[int] = None,
        notify_user_id: Optional[uuid.UUID] = None,
        skip_local_validation: bool = False,
        **_ignored,
    ) -> Appointment:
        """
        Write a local Appointment row. For Calendly-connected tenants this is a
        read-log of a slot already validated/scheduled on Calendly (pass
        skip_local_validation=True to skip the redundant local check). For the
        local web flow and the legacy (non-Calendly) voice path, this is the
        only conflict guard, so past-date and overlap checks still run here.
        """
        resolved_duration = duration_minutes or DEFAULT_APPOINTMENT_DURATION_MINUTES
        slot_start_utc = _ensure_utc(slot_start)
        slot_end_utc = slot_start_utc + timedelta(minutes=resolved_duration)

        if not skip_local_validation:
            self._validate_local_slot_bookable(slot_start_utc, slot_end_utc, db, tenant_id)

        appt = Appointment(
            tenant_id=tenant_id,
            agent_id=agent_id,
            call_session_id=call_session_id,
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            appointment_reason=appointment_reason,
            slot_start=slot_start_utc,
            slot_end=slot_end_utc,
            duration_minutes=resolved_duration,
            status="pending",
            created_via=created_via,
            notes=notes,
        )
        db.add(appt)
        try:
            db.commit()
            db.refresh(appt)
            logger.info(
                "Appointment logged: tenant=%s agent=%s slot=%s customer=%s",
                tenant_id, agent_id, slot_start_utc, customer_name,
            )
        except IntegrityError:
            db.rollback()
            raise ValueError(
                f"The {_fmt_slot_label(slot_start_utc)} slot could not be logged. "
                "Please try again."
            )
        self._send_appointment_confirmation_email(
            db=db,
            tenant_id=tenant_id,
            appt=appt,
            notify_user_id=notify_user_id,
            call_session_id=call_session_id,
        )
        return appt

    def get_active_appointment_for_call_session(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        call_session_id: uuid.UUID,
    ) -> Optional[Appointment]:
        """Latest confirmed/pending appointment tied to this call (for in-call reschedule)."""
        return (
            db.query(Appointment)
            .filter(
                Appointment.tenant_id == tenant_id,
                Appointment.call_session_id == call_session_id,
                Appointment.status.in_(["confirmed", "pending"]),
            )
            .order_by(Appointment.created_at.desc())
            .first()
        )

    def reschedule_appointment(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        appointment_id: uuid.UUID,
        slot_start: datetime,
        customer_name: Optional[str] = None,
        customer_phone: Optional[str] = None,
        appointment_reason: Optional[str] = None,
        customer_email: Optional[str] = None,
        notes: Optional[str] = None,
        duration_minutes: Optional[int] = None,
        notify_user_id: Optional[uuid.UUID] = None,
        skip_local_validation: bool = False,
    ) -> Appointment:
        """
        Move an existing appointment to a new slot. For Calendly-connected
        tenants Calendly already validated the new slot (pass
        skip_local_validation=True); otherwise this is the only guard against
        overlaps/past-dated slots for the local record.
        """
        appt = self.get_appointment_by_id(db, appointment_id, tenant_id)
        if not appt:
            raise ValueError("Appointment not found.")
        if appt.status not in ("confirmed", "pending"):
            raise ValueError(
                f"Cannot reschedule an appointment that is {appt.status}."
            )

        resolved_duration = (
            duration_minutes if duration_minutes is not None else appt.duration_minutes
        )
        slot_start_utc = _ensure_utc(slot_start)
        slot_end_utc = slot_start_utc + timedelta(minutes=resolved_duration)

        if not skip_local_validation:
            self._validate_local_slot_bookable(
                slot_start_utc, slot_end_utc, db, tenant_id,
                exclude_appointment_id=appointment_id,
            )

        if customer_name is not None:
            appt.customer_name = customer_name
        if customer_phone is not None:
            appt.customer_phone = customer_phone
        if appointment_reason is not None:
            appt.appointment_reason = appointment_reason
        if customer_email is not None:
            appt.customer_email = customer_email
        if notes is not None:
            appt.notes = notes

        appt.slot_start = slot_start_utc
        appt.slot_end = slot_end_utc
        appt.duration_minutes = resolved_duration

        try:
            db.commit()
            db.refresh(appt)
            logger.info(
                "Appointment rescheduled: id=%s tenant=%s new_slot=%s",
                appointment_id,
                tenant_id,
                slot_start_utc,
            )
        except IntegrityError:
            db.rollback()
            raise ValueError(
                f"The {_fmt_slot_label(slot_start_utc)} slot could not be logged. "
                "Please try again."
            )
        self._send_appointment_confirmation_email(
            db=db,
            tenant_id=tenant_id,
            appt=appt,
            notify_user_id=notify_user_id,
            call_session_id=appt.call_session_id,
        )
        if appt.status == "confirmed":
            from app.services.appointment_follow_up_service import refresh_follow_up_crm_after_reschedule

            refresh_follow_up_crm_after_reschedule(db, appt, notify_user_id)
        return appt

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_appointments(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        agent_id: Optional[uuid.UUID] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[Appointment], int]:
        """Returns (appointments_page, total_count)."""
        q = db.query(Appointment).filter(Appointment.tenant_id == tenant_id)
        if date_from:
            q = q.filter(
                Appointment.slot_start >= datetime.combine(date_from, dt_time.min, tzinfo=timezone.utc)
            )
        if date_to:
            q = q.filter(
                Appointment.slot_start <= datetime.combine(date_to, dt_time.max, tzinfo=timezone.utc)
            )
        if agent_id:
            q = q.filter(Appointment.agent_id == agent_id)
        if status:
            q = q.filter(Appointment.status == status)
        total = q.count()
        items = q.order_by(Appointment.slot_start.asc()).offset(offset).limit(limit).all()
        return items, total

    def get_appointment_by_id(
        self, db: Session, appointment_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> Optional[Appointment]:
        return (
            db.query(Appointment)
            .filter(Appointment.id == appointment_id, Appointment.tenant_id == tenant_id)
            .first()
        )

    def appointment_local_display(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        appt: Appointment,
    ) -> Tuple[str, datetime, datetime]:
        """
        Same instants as slot_start/slot_end (UTC in DB). Local display timezone
        is now resolved by Calendly at booking time, so this returns UTC.
        """
        _ = (db, tenant_id)
        utc_start = _ensure_utc(appt.slot_start)
        utc_end = _ensure_utc(appt.slot_end)
        return ("UTC", utc_start, utc_end)

    def to_appointment_out(self, db: Session, tenant_id: uuid.UUID, appt: Appointment) -> AppointmentOut:
        """Build AppointmentOut with UTC slot_* plus additive local fields for display."""
        tz_label, start_l, end_l = self.appointment_local_display(db, tenant_id, appt)
        base = AppointmentOut.model_validate(appt)
        return base.model_copy(
            update={
                "business_timezone": tz_label,
                "slot_start_local": start_l,
                "slot_end_local": end_l,
            }
        )

    def update_appointment_status(
        self,
        db: Session,
        appointment_id: uuid.UUID,
        tenant_id: uuid.UUID,
        status: str,
        cancellation_reason: Optional[str] = None,
        notes: Optional[str] = None,
        notify_user_id: Optional[uuid.UUID] = None,
    ) -> Optional[Appointment]:
        appt = self.get_appointment_by_id(db, appointment_id, tenant_id)
        if not appt:
            return None

        allowed = ALLOWED_STATUS_TRANSITIONS.get(appt.status, set())
        if status not in allowed:
            raise ValueError(
                f"Cannot transition from '{appt.status}' to '{status}'. "
                f"Allowed transitions: {', '.join(sorted(allowed)) or 'none (terminal state)'}."
            )

        was_confirmed = appt.status == "confirmed"
        appt.status = status
        if cancellation_reason is not None:
            appt.cancellation_reason = cancellation_reason
        if notes is not None:
            appt.notes = notes
        if (not was_confirmed) and appt.status == "confirmed":
            self._notify_customer_confirmation_if_needed(
                db=db,
                tenant_id=tenant_id,
                appt=appt,
            )
        db.commit()
        db.refresh(appt)
        if (not was_confirmed) and appt.status == "confirmed":
            from app.services.appointment_follow_up_service import schedule_follow_up_after_confirm

            schedule_follow_up_after_confirm(db, appt, notify_user_id)
        if appt.status == "cancelled":
            from app.services.appointment_follow_up_service import cancel_follow_up_crm_card

            cancel_follow_up_crm_card(db, appt, notify_user_id)
        return appt

    def delete_appointment(
        self, db: Session, appointment_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> bool:
        appt = self.get_appointment_by_id(db, appointment_id, tenant_id)
        if not appt:
            return False
        from app.services.appointment_follow_up_service import cancel_follow_up_crm_card

        cancel_follow_up_crm_card(db, appt, None)
        appt = self.get_appointment_by_id(db, appointment_id, tenant_id)
        if not appt:
            return False
        db.delete(appt)
        db.commit()
        return True


calendar_service = CalendarService()
