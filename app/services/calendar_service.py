"""
Calendar Service
Handles business hours, blocked slots, and appointment booking logic.
All operations are scoped to tenant_id for multi-tenant isolation.
"""
import html
from datetime import datetime, date, timedelta, timezone, time as dt_time, tzinfo
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import uuid

from app.core.config import settings
from app.core.logger import logger
from app.models.business_hours import BusinessHours
from app.models.blocked_slot import BlockedSlot
from app.models.appointment import Appointment
from app.models.slot_reservation import SlotReservation
from app.models.call_session import CallSession
from app.models.user import User
from app.schemas.calendar import (
    AppointmentOut,
    BusinessHoursUpsert,
    BlockedSlotCreate,
    AvailableSlot,
    AvailableSlotsResponse,
)
from app.services.email_service import email_service
from app.utils.spoken_email import normalize_stored_email

SLOT_BOOKING_BUFFER_MINUTES = 15
APPOINTMENT_REVIEW_TOKEN_TTL_HOURS = 24 * 7


class BusinessHoursConflictError(Exception):
    """Raised when creating business hours for weekdays that already exist for the tenant."""

    def __init__(self, days: List[int]):
        self.days = days
        super().__init__()


ALLOWED_STATUS_TRANSITIONS = {
    "pending":   {"confirmed", "cancelled"},
    "confirmed": {"completed", "cancelled", "no_show"},
    "cancelled": set(),
    "completed": set(),
    "no_show":   set(),
}


def _safe_tz(tz_str: str) -> tzinfo:
    try:
        return ZoneInfo(tz_str)
    except (ZoneInfoNotFoundError, Exception):
        return timezone.utc


def _parse_time_str(t: str) -> Optional[dt_time]:
    """Parse 'HH:MM' string into a time object."""
    try:
        h, m = t.split(":")
        return dt_time(int(h), int(m))
    except Exception:
        return None


def _fmt_slot_label(dt: datetime) -> str:
    """Format datetime to '9:00 AM' (no leading zero)."""
    return dt.strftime("%I:%M %p").lstrip("0") or "12:00 AM"


def _ensure_utc(dt_val: datetime) -> datetime:
    """Normalise a datetime to UTC. Treats naive datetimes as UTC."""
    if dt_val.tzinfo is None:
        return dt_val.replace(tzinfo=timezone.utc)
    return dt_val.astimezone(timezone.utc)


class CalendarService:

    # ── Internal validation ───────────────────────────────────────────────────

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

    def _get_tenant_tz(self, db: Session, tenant_id: uuid.UUID) -> tzinfo:
        return _safe_tz(self.get_tenant_timezone(db, tenant_id))

    def _localize_input_datetime(
        self, db: Session, tenant_id: uuid.UUID, dt_val: datetime
    ) -> datetime:
        tenant_tz = self._get_tenant_tz(db, tenant_id)
        if dt_val.tzinfo is None:
            return dt_val.replace(tzinfo=tenant_tz)
        return dt_val.astimezone(tenant_tz)

    def _resolve_booking_context(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        slot_start: datetime,
        duration_minutes: Optional[int] = None,
    ) -> tuple[BusinessHours, tzinfo, datetime, datetime, datetime, datetime, int]:
        slot_local_seed = self._localize_input_datetime(db, tenant_id, slot_start)
        bh = self._get_business_hours_for_date(db, tenant_id, slot_local_seed.date())

        if not bh or bh.is_closed or not bh.open_time or not bh.close_time:
            raise ValueError(
                f"The business is closed on {slot_local_seed.strftime('%A')}. "
                "Please choose another day."
            )

        tz_info = _safe_tz(bh.timezone)
        if slot_start.tzinfo is None:
            slot_local = slot_start.replace(tzinfo=tz_info)
        else:
            slot_local = slot_start.astimezone(tz_info)

        resolved_duration = duration_minutes or bh.slot_duration_minutes
        slot_end_local = slot_local + timedelta(minutes=resolved_duration)
        slot_start_utc = slot_local.astimezone(timezone.utc)
        slot_end_utc = slot_end_local.astimezone(timezone.utc)

        return (
            bh,
            tz_info,
            slot_local,
            slot_end_local,
            slot_start_utc,
            slot_end_utc,
            resolved_duration,
        )

    def _get_overlapping_appointment(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        slot_start: datetime,
        slot_end: datetime,
        exclude_appointment_id: Optional[uuid.UUID] = None,
    ) -> Optional[Appointment]:
        q = (
            db.query(Appointment)
            .filter(
                Appointment.tenant_id == tenant_id,
                Appointment.status.notin_(["cancelled"]),
                Appointment.slot_start < slot_end,
                Appointment.slot_end > slot_start,
            )
        )
        if exclude_appointment_id is not None:
            q = q.filter(Appointment.id != exclude_appointment_id)
        return q.order_by(Appointment.slot_start.asc()).first()

    def _get_overlapping_reservation(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        slot_start: datetime,
        slot_end: datetime,
        exclude_reservation_id: Optional[uuid.UUID] = None,
    ) -> Optional[SlotReservation]:
        q = (
            db.query(SlotReservation)
            .filter(
                SlotReservation.tenant_id == tenant_id,
                SlotReservation.status == "active",
                SlotReservation.slot_start < slot_end,
                SlotReservation.slot_end > slot_start,
            )
        )
        if exclude_reservation_id is not None:
            q = q.filter(SlotReservation.id != exclude_reservation_id)
        return q.order_by(SlotReservation.slot_start.asc()).first()

    def resolve_slot_window(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        slot_start: datetime,
        duration_minutes: Optional[int] = None,
    ) -> tuple[BusinessHours, tzinfo, datetime, datetime, datetime, datetime, int]:
        """
        Public wrapper for booking time-window resolution (same as internal booking path).
        """
        return self._resolve_booking_context(
            db=db,
            tenant_id=tenant_id,
            slot_start=slot_start,
            duration_minutes=duration_minutes,
        )

    def _validate_slot_bookable(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        slot_local: datetime,
        slot_end_local: datetime,
        slot_start_utc: datetime,
        slot_end_utc: datetime,
        bh: BusinessHours,
        tz_info: tzinfo,
    ) -> None:
        """
        Raises ValueError if the slot cannot be booked:
        past/too-soon, off-grid, outside business hours, or inside a blocked range.
        """
        now_utc = datetime.now(timezone.utc)
        buffer = timedelta(minutes=SLOT_BOOKING_BUFFER_MINUTES)

        if slot_start_utc <= now_utc + buffer:
            raise ValueError(
                "Cannot book a slot in the past or within the next "
                f"{SLOT_BOOKING_BUFFER_MINUTES} minutes. Please choose a later time."
            )

        opening_dt = datetime.combine(slot_local.date(), bh.open_time, tzinfo=tz_info)
        closing_dt = datetime.combine(slot_local.date(), bh.close_time, tzinfo=tz_info)

        if slot_local.second or slot_local.microsecond:
            raise ValueError("Appointments must start on an exact minute boundary.")

        minutes_from_open = int((slot_local - opening_dt).total_seconds() // 60)
        if slot_local < opening_dt or slot_end_local > closing_dt:
            raise ValueError(
                f"The slot is outside business hours "
                f"({bh.open_time.strftime('%H:%M')} – {bh.close_time.strftime('%H:%M')})."
            )

        if minutes_from_open < 0 or minutes_from_open % bh.slot_duration_minutes != 0:
            raise ValueError(
                f"Appointments must start on the configured "
                f"{bh.slot_duration_minutes}-minute slot boundaries."
            )

        blocked = (
            db.query(BlockedSlot)
            .filter(
                BlockedSlot.tenant_id == tenant_id,
                BlockedSlot.blocked_from < slot_end_utc,
                BlockedSlot.blocked_until > slot_start_utc,
            )
            .first()
        )
        if blocked:
            raise ValueError(
                f"This time slot is blocked ({blocked.title}). Please choose another time."
            )

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

        return appt

    # ── Slot availability ─────────────────────────────────────────────────────

    def get_available_slots(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        target_date: date,
        agent_id: Optional[uuid.UUID] = None,
    ) -> AvailableSlotsResponse:
        """
        Return bookable slots for a given date.
        Automatically excludes: past slots (with buffer), blocked slots, already-booked slots.
        """
        # Availability is tenant-wide. Keep the parameter for backward compatibility.
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
        blocked = (
            db.query(BlockedSlot)
            .filter(
                BlockedSlot.tenant_id == tenant_id,
                BlockedSlot.blocked_from < day_end,
                BlockedSlot.blocked_until > day_start,
            )
            .all()
        )
        blocked_ranges = [
            (_ensure_utc(item.blocked_from), _ensure_utc(item.blocked_until))
            for item in blocked
        ]

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

        resv = (
            db.query(SlotReservation.slot_start, SlotReservation.slot_end)
            .filter(
                SlotReservation.tenant_id == tenant_id,
                SlotReservation.status == "active",
                SlotReservation.slot_start < day_end,
                SlotReservation.slot_end > day_start,
            )
            .all()
        )
        reserved_ranges = [(_ensure_utc(bs), _ensure_utc(be)) for bs, be in resv]

        available: List[AvailableSlot] = []
        for s_start, s_end in all_slots:
            s_utc = s_start.astimezone(timezone.utc)
            s_end_utc = s_end.astimezone(timezone.utc)

            if s_utc <= now_utc + buffer:
                continue

            if any(blocked_from < s_end_utc and blocked_until > s_utc for blocked_from, blocked_until in blocked_ranges):
                continue

            if any(bs < s_end_utc and be > s_utc for bs, be in booked_ranges):
                continue

            if any(rs < s_end_utc and re_ > s_utc for rs, re_ in reserved_ranges):
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

    # ── Booking ───────────────────────────────────────────────────────────────

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
        consuming_reservation_id: Optional[uuid.UUID] = None,
    ) -> Appointment:
        """
        Book a slot. Raises ValueError if the slot is unavailable, in the past,
        outside business hours, or blocked.
        Uses DB-level unique constraints as the final guard against races.
        `consuming_reservation_id` excludes that active hold when checking conflicts
        (used when finalizing a voice in-call reservation post-call).
        """
        (
            bh,
            tz_info,
            slot_local,
            slot_end_local,
            slot_start_utc,
            slot_end_utc,
            resolved_duration,
        ) = self._resolve_booking_context(
            db=db,
            tenant_id=tenant_id,
            slot_start=slot_start,
            duration_minutes=duration_minutes,
        )

        self._validate_slot_bookable(
            db=db,
            tenant_id=tenant_id,
            slot_local=slot_local,
            slot_end_local=slot_end_local,
            slot_start_utc=slot_start_utc,
            slot_end_utc=slot_end_utc,
            bh=bh,
            tz_info=tz_info,
        )

        conflict = self._get_overlapping_appointment(
            db=db,
            tenant_id=tenant_id,
            slot_start=slot_start_utc,
            slot_end=slot_end_utc,
        )
        if conflict:
            raise ValueError(
                f"The {_fmt_slot_label(slot_local)} slot is no longer available. "
                "Please choose another time."
            )

        res_hold = self._get_overlapping_reservation(
            db=db,
            tenant_id=tenant_id,
            slot_start=slot_start_utc,
            slot_end=slot_end_utc,
            exclude_reservation_id=consuming_reservation_id,
        )
        if res_hold:
            raise ValueError(
                f"The {_fmt_slot_label(slot_local)} slot is no longer available. "
                "Please choose another time."
            )

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
                "Appointment booked: tenant=%s agent=%s slot=%s customer=%s",
                tenant_id, agent_id, slot_start_utc, customer_name,
            )
        except IntegrityError:
            db.rollback()
            raise ValueError(
                f"The {_fmt_slot_label(slot_local)} slot was just taken. "
                "Please choose another time."
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
    ) -> Appointment:
        """
        Move an existing appointment to a new slot. Same validation as booking;
        the current appointment is excluded from overlap checks.
        """
        appt = self.get_appointment_by_id(db, appointment_id, tenant_id)
        if not appt:
            raise ValueError("Appointment not found.")
        if appt.status not in ("confirmed", "pending"):
            raise ValueError(
                f"Cannot reschedule an appointment that is {appt.status}."
            )

        eff_duration = (
            duration_minutes if duration_minutes is not None else appt.duration_minutes
        )

        (
            bh,
            tz_info,
            slot_local,
            slot_end_local,
            slot_start_utc,
            slot_end_utc,
            resolved_duration,
        ) = self._resolve_booking_context(
            db=db,
            tenant_id=tenant_id,
            slot_start=slot_start,
            duration_minutes=eff_duration,
        )

        self._validate_slot_bookable(
            db=db,
            tenant_id=tenant_id,
            slot_local=slot_local,
            slot_end_local=slot_end_local,
            slot_start_utc=slot_start_utc,
            slot_end_utc=slot_end_utc,
            bh=bh,
            tz_info=tz_info,
        )

        conflict = self._get_overlapping_appointment(
            db=db,
            tenant_id=tenant_id,
            slot_start=slot_start_utc,
            slot_end=slot_end_utc,
            exclude_appointment_id=appointment_id,
        )
        if conflict:
            raise ValueError(
                f"The {_fmt_slot_label(slot_local)} slot is no longer available. "
                "Please choose another time."
            )

        res_hold = self._get_overlapping_reservation(
            db=db,
            tenant_id=tenant_id,
            slot_start=slot_start_utc,
            slot_end=slot_end_utc,
        )
        if res_hold:
            raise ValueError(
                f"The {_fmt_slot_label(slot_local)} slot is no longer available. "
                "Please choose another time."
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
                f"The {_fmt_slot_label(slot_local)} slot was just taken. "
                "Please choose another time."
            )
        self._send_appointment_confirmation_email(
            db=db,
            tenant_id=tenant_id,
            appt=appt,
            notify_user_id=notify_user_id,
            call_session_id=appt.call_session_id,
        )
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
        Same instants as slot_start/slot_end (UTC in DB), expressed in the business-hours
        timezone for the appointment's local calendar day. Used for additive API fields only.
        """
        utc_start = _ensure_utc(appt.slot_start)
        utc_end = _ensure_utc(appt.slot_end)
        tenant_tz_str = self.get_tenant_timezone(db, tenant_id)
        tenant_tz = _safe_tz(tenant_tz_str)
        local_seed = utc_start.astimezone(tenant_tz)
        bh = self._get_business_hours_for_date(db, tenant_id, local_seed.date())
        if bh and bh.timezone:
            tz_info = _safe_tz(bh.timezone)
            tz_label = bh.timezone
        else:
            tz_info = tenant_tz
            tz_label = tenant_tz_str
        return (tz_label, utc_start.astimezone(tz_info), utc_end.astimezone(tz_info))

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
        return appt

    def delete_appointment(
        self, db: Session, appointment_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> bool:
        appt = self.get_appointment_by_id(db, appointment_id, tenant_id)
        if not appt:
            return False
        db.delete(appt)
        db.commit()
        return True

    # ── Business Hours ────────────────────────────────────────────────────────

    def get_business_hours(self, db: Session, tenant_id: uuid.UUID) -> List[BusinessHours]:
        return (
            db.query(BusinessHours)
            .filter(
                BusinessHours.tenant_id == tenant_id,
                BusinessHours.is_deleted.is_(False),
            )
            .order_by(BusinessHours.day_of_week.asc())
            .all()
        )

    def get_tenant_timezone(self, db: Session, tenant_id: uuid.UUID) -> str:
        """Return the timezone string from the first configured business-hours row."""
        bh = (
            db.query(BusinessHours.timezone)
            .filter(
                BusinessHours.tenant_id == tenant_id,
                BusinessHours.is_deleted.is_(False),
            )
            .first()
        )
        return bh[0] if bh else "UTC"

    def create_business_hours(
        self, db: Session, tenant_id: uuid.UUID, hours_list: List[BusinessHoursUpsert]
    ) -> List[BusinessHours]:
        """Insert business hours only; fails if any requested weekday already exists."""
        if not hours_list:
            return []
        days = [item.day_of_week for item in hours_list]
        if len(days) != len(set(days)):
            raise ValueError("Duplicate day_of_week values in request body.")
        existing = (
            db.query(BusinessHours)
            .filter(
                BusinessHours.tenant_id == tenant_id,
                BusinessHours.day_of_week.in_(days),
            )
            .all()
        )
        active_existing_days = sorted({row.day_of_week for row in existing if not row.is_deleted})
        if active_existing_days:
            raise BusinessHoursConflictError(active_existing_days)

        deleted_by_day = {row.day_of_week: row for row in existing if row.is_deleted}

        results: List[BusinessHours] = []
        for item in hours_list:
            open_t = _parse_time_str(item.open_time) if item.open_time else None
            close_t = _parse_time_str(item.close_time) if item.close_time else None
            bh = deleted_by_day.get(item.day_of_week)
            if bh:
                bh.open_time = open_t
                bh.close_time = close_t
                bh.is_closed = item.is_closed
                bh.timezone = item.timezone
                bh.slot_duration_minutes = item.slot_duration_minutes
                bh.is_deleted = False
                bh.deleted_at = None
            else:
                bh = BusinessHours(
                    tenant_id=tenant_id,
                    day_of_week=item.day_of_week,
                    open_time=open_t,
                    close_time=close_t,
                    is_closed=item.is_closed,
                    timezone=item.timezone,
                    slot_duration_minutes=item.slot_duration_minutes,
                    is_deleted=False,
                    deleted_at=None,
                )
                db.add(bh)
            results.append(bh)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            raise BusinessHoursConflictError(days)
        for r in results:
            db.refresh(r)
        return results

    def upsert_business_hours(
        self, db: Session, tenant_id: uuid.UUID, hours_list: List[BusinessHoursUpsert]
    ) -> List[BusinessHours]:
        results = []
        for item in hours_list:
            existing = (
                db.query(BusinessHours)
                .filter(
                    BusinessHours.tenant_id == tenant_id,
                    BusinessHours.day_of_week == item.day_of_week,
                )
                .first()
            )
            open_t = _parse_time_str(item.open_time) if item.open_time else None
            close_t = _parse_time_str(item.close_time) if item.close_time else None
            if existing:
                existing.open_time = open_t
                existing.close_time = close_t
                existing.is_closed = item.is_closed
                existing.timezone = item.timezone
                existing.slot_duration_minutes = item.slot_duration_minutes
                existing.is_deleted = False
                existing.deleted_at = None
                results.append(existing)
            else:
                bh = BusinessHours(
                    tenant_id=tenant_id,
                    day_of_week=item.day_of_week,
                    open_time=open_t,
                    close_time=close_t,
                    is_closed=item.is_closed,
                    timezone=item.timezone,
                    slot_duration_minutes=item.slot_duration_minutes,
                    is_deleted=False,
                    deleted_at=None,
                )
                db.add(bh)
                results.append(bh)
        db.commit()
        for r in results:
            db.refresh(r)
        return results

    def delete_business_hours(
        self, db: Session, business_hours_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> bool:
        row = (
            db.query(BusinessHours)
            .filter(
                BusinessHours.id == business_hours_id,
                BusinessHours.tenant_id == tenant_id,
            )
            .first()
        )
        if not row:
            return False
        row.is_deleted = True
        row.deleted_at = datetime.now(timezone.utc)
        db.commit()
        return True

    # ── Blocked Slots ─────────────────────────────────────────────────────────

    def get_blocked_slots(self, db: Session, tenant_id: uuid.UUID) -> List[BlockedSlot]:
        return (
            db.query(BlockedSlot)
            .filter(BlockedSlot.tenant_id == tenant_id)
            .order_by(BlockedSlot.blocked_from.asc())
            .all()
        )

    def create_blocked_slot(
        self, db: Session, tenant_id: uuid.UUID, data: BlockedSlotCreate
    ) -> BlockedSlot:
        blocked_from = _ensure_utc(self._localize_input_datetime(db, tenant_id, data.blocked_from))
        blocked_until = _ensure_utc(self._localize_input_datetime(db, tenant_id, data.blocked_until))

        bs = BlockedSlot(
            tenant_id=tenant_id,
            title=data.title,
            blocked_from=blocked_from,
            blocked_until=blocked_until,
        )
        db.add(bs)
        db.commit()
        db.refresh(bs)
        return bs

    def delete_blocked_slot(
        self, db: Session, blocked_slot_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> bool:
        bs = (
            db.query(BlockedSlot)
            .filter(BlockedSlot.id == blocked_slot_id, BlockedSlot.tenant_id == tenant_id)
            .first()
        )
        if not bs:
            return False
        db.delete(bs)
        db.commit()
        return True


calendar_service = CalendarService()
