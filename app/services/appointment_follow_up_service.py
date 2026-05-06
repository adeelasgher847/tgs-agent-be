"""
Appointment confirmation → Trello scheduled reminder call (follow-up agent), 2h before slot (UTC).
Does not modify resume interview flows.
"""

from __future__ import annotations

import html
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logger import logger
from app.models.agent import Agent
from app.models.appointment import Appointment
from app.models.call_session import CallSession
from app.models.phone_number import PhoneNumber
from app.models.scheduled_call import ScheduledCall
from app.models.tenant_crm_config import CRMConfig
from app.models.user import User, user_tenant_association
from app.services.agent_service import agent_service
from app.services.billing_service import BillingService
from app.services.scheduled_call_service import ScheduledCallService
from app.services.trello_service import TrelloService
from app.services.crm_config_service import CRMConfigService
from app.services.crm_service_factory import CRMServiceFactory
from app.services.email_service import email_service

FOLLOWUP_LEAD = timedelta(hours=2)
LAST_MINUTE_BUFFER = timedelta(minutes=2)


def send_follow_up_outcome_staff_email(
    db: Session,
    *,
    staff_user_id: Optional[uuid.UUID],
    tenant_id: uuid.UUID,
    appointment_id: uuid.UUID,
    outcome: str,
    detail: str = "",
) -> None:
    """Notify tenant staff (account user) about follow-up call outcome."""
    try:
        uid = resolve_acting_user_id_for_follow_up(db, tenant_id, staff_user_id)
        if not uid:
            return
        user = db.query(User).filter(User.id == uid).first()
        if not user or not (user.email or "").strip():
            return
        subj = f"Assistly | Appointment follow-up: {outcome.replace('_', ' ')}"
        body = (
            f"<p><strong>Outcome:</strong> {html.escape(outcome)}</p>"
            f"<p><strong>Appointment ID:</strong> {appointment_id}</p>"
            f"<p><strong>Tenant ID:</strong> {tenant_id}</p>"
        )
        if detail:
            body += f"<p><strong>Detail:</strong> {html.escape(detail[:2000])}</p>"
        email_service.send_generic_email(
            to_email=user.email.strip(),
            subject=subj,
            html_body=body,
        )
    except Exception:
        logger.exception("Follow-up staff email failed appt=%s", appointment_id)


def resolve_trello_crm_config_id_for_user(db: Session, user_id: uuid.UUID) -> Optional[uuid.UUID]:
    """Trello-first CRM config for scheduled-call board (mirrors resume flow; resume routes unchanged)."""
    trello_link = (
        db.query(ScheduledCall)
        .filter(
            ScheduledCall.user_id == user_id,
            ScheduledCall.tenant_crm_config_id.isnot(None),
            ScheduledCall.crm_type == "trello",
        )
        .order_by(ScheduledCall.created_at.desc())
        .first()
    )
    if trello_link and trello_link.tenant_crm_config_id:
        return trello_link.tenant_crm_config_id

    any_link = (
        db.query(ScheduledCall)
        .filter(
            ScheduledCall.user_id == user_id,
            ScheduledCall.tenant_crm_config_id.isnot(None),
        )
        .order_by(ScheduledCall.created_at.desc())
        .first()
    )
    if any_link and any_link.crm_type == "trello" and any_link.tenant_crm_config_id:
        return any_link.tenant_crm_config_id

    global_trello = db.query(CRMConfig).filter(CRMConfig.crm_type == "trello").first()
    if global_trello:
        return global_trello.id
    return None


def resolve_acting_user_id_for_follow_up(
    db: Session,
    tenant_id: uuid.UUID,
    preferred: Optional[uuid.UUID],
) -> Optional[uuid.UUID]:
    if preferred:
        u = db.query(User).filter(User.id == preferred).first()
        if u:
            return u.id
    creator = db.execute(
        select(user_tenant_association.c.user_id)
        .where(
            user_tenant_association.c.tenant_id == tenant_id,
            user_tenant_association.c.is_creator.is_(True),
        )
        .limit(1)
    ).scalar_one_or_none()
    if creator:
        return creator
    fallback = (
        db.query(User)
        .filter(User.current_tenant_id == tenant_id)
        .order_by(User.created_at.asc())
        .first()
    )
    return fallback.id if fallback else None


def _reminder_time_utc(slot_start_utc: datetime) -> datetime:
    if slot_start_utc.tzinfo is None:
        slot_start_utc = slot_start_utc.replace(tzinfo=timezone.utc)
    else:
        slot_start_utc = slot_start_utc.astimezone(timezone.utc)
    reminder = slot_start_utc - FOLLOWUP_LEAD
    now = datetime.now(timezone.utc)
    if reminder <= now:
        reminder = now + LAST_MINUTE_BUFFER
    return reminder


def _follow_up_phone_number_id(db: Session, tenant_id: uuid.UUID, follow_agent: Agent) -> Optional[str]:
    pn = (
        db.query(PhoneNumber)
        .filter(
            PhoneNumber.assistant_id == follow_agent.id,
            PhoneNumber.tenant_id == tenant_id,
            PhoneNumber.status == "active",
        )
        .first()
    )
    return str(pn.id) if pn else None


def _resolve_phone_number_id_from_appointment_origin(
    db: Session,
    *,
    appt: Appointment,
    follow_agent: Agent,
) -> Optional[str]:
    """
    Prefer the tenant phone number used in the original appointment call/session.
    Fallback to follow-up agent assigned active number.
    """
    # 1) Try to resolve from appointment.call_session_id by matching tenant-owned numbers.
    if appt.call_session_id:
        cs = (
            db.query(CallSession)
            .filter(
                CallSession.id == appt.call_session_id,
                CallSession.tenant_id == appt.tenant_id,
            )
            .first()
        )
        if cs:
            # For inbound: tenant number is usually `to_number`.
            # For outbound: tenant number is usually `from_number`.
            # Keep robust fallback order for legacy/mixed records.
            candidate_numbers = [
                (cs.to_number or "").strip(),
                (cs.from_number or "").strip(),
                (cs.assistant_phone_number or "").strip(),
            ]
            for num in candidate_numbers:
                if not num:
                    continue
                pn = (
                    db.query(PhoneNumber)
                    .filter(
                        PhoneNumber.tenant_id == appt.tenant_id,
                        PhoneNumber.phone_number == num,
                        PhoneNumber.status == "active",
                    )
                    .first()
                )
                if pn:
                    return str(pn.id)

    # 2) Fallback: follow-up agent assigned active number.
    return _follow_up_phone_number_id(db, appt.tenant_id, follow_agent)


def schedule_follow_up_after_confirm(
    db: Session,
    appt: Appointment,
    acting_user_id: Optional[uuid.UUID],
) -> None:
    """
    After appointment is confirmed: create Trello scheduled call row for follow-up agent.
    Safe to call from API paths; logs and returns on any skip/failure.
    """
    try:
        if appt.status != "confirmed":
            return
        if getattr(appt, "follow_up_crm_item_id", None):
            return

        follow = agent_service.get_follow_up_agent_by_tenant(db, appt.tenant_id)
        if not follow:
            return

        user_id = resolve_acting_user_id_for_follow_up(db, appt.tenant_id, acting_user_id)
        if not user_id:
            logger.warning(
                "Follow-up schedule skipped: no acting user for tenant=%s appt=%s",
                appt.tenant_id,
                appt.id,
            )
            return

        crm_id = resolve_trello_crm_config_id_for_user(db, user_id)
        if not crm_id:
            logger.info(
                "Follow-up schedule skipped: no Trello CRM config for user=%s appt=%s",
                user_id,
                appt.id,
            )
            return

        if not BillingService.has_crm_access(db, user_id, "trello"):
            logger.info(
                "Follow-up schedule skipped: no Trello subscription for user=%s appt=%s",
                user_id,
                appt.id,
            )
            return

        phone = (appt.customer_phone or "").strip()
        if not phone.startswith("+"):
            logger.warning(
                "Follow-up schedule skipped: customer_phone must be E.164 with + appt=%s",
                appt.id,
            )
            return

        crm_cfg = CRMConfigService().get_crm_config_by_id(db, crm_id)
        if not crm_cfg or crm_cfg.crm_type != "trello":
            logger.info("Follow-up schedule skipped: CRM config is not trello appt=%s", appt.id)
            return

        reminder = _reminder_time_utc(appt.slot_start)
        phone_number_id = _resolve_phone_number_id_from_appointment_origin(
            db,
            appt=appt,
            follow_agent=follow,
        )
        jd_context: Dict[str, Any] = {"appointment_id": str(appt.id)}

        result = ScheduledCallService.create_single_scheduled_call_sync(
            db=db,
            tenant_id=appt.tenant_id,
            user_id=user_id,
            phone_number=phone,
            agent_id=follow.id,
            call_time_utc=reminder.isoformat(),
            crm_config_id=crm_id,
            phone_number_id=phone_number_id,
            jd_context=jd_context,
        )
        item_id = result.get("item_id")
        if item_id:
            appt.follow_up_crm_item_id = str(item_id)
            db.add(appt)
            db.commit()
            logger.info(
                "Follow-up Trello item created appt=%s item=%s call_time=%s",
                appt.id,
                item_id,
                reminder.isoformat(),
            )
    except Exception:
        logger.exception(
            "Follow-up scheduling failed appt=%s tenant=%s",
            getattr(appt, "id", None),
            getattr(appt, "tenant_id", None),
        )
        try:
            db.rollback()
        except Exception:
            pass


def _trello_board_and_map(
    db: Session,
    *,
    user_id: uuid.UUID,
    tenant_id: uuid.UUID,
    crm_config_id: uuid.UUID,
) -> tuple[Any, Dict[str, str]]:
    board_record, field_map = ScheduledCallService.get_or_create_board_for_user(
        db, user_id, tenant_id, crm_config_id
    )
    return board_record, field_map


def refresh_follow_up_crm_after_reschedule(
    db: Session,
    appt: Appointment,
    acting_user_id: Optional[uuid.UUID],
) -> None:
    """Move follow-up reminder on Trello when appointment slot changes."""
    if not appt.follow_up_crm_item_id:
        return
    try:
        user_id = resolve_acting_user_id_for_follow_up(db, appt.tenant_id, acting_user_id)
        if not user_id:
            return
        crm_id = resolve_trello_crm_config_id_for_user(db, user_id)
        if not crm_id:
            return
        board_record, field_map = _trello_board_and_map(
            db, user_id=user_id, tenant_id=appt.tenant_id, crm_config_id=crm_id
        )
        if board_record.crm_type != "trello":
            return
        crm = CRMServiceFactory.get_service(CRMConfigService().get_crm_config_by_id(db, crm_id))
        if not isinstance(crm, TrelloService):
            return
        reminder = _reminder_time_utc(appt.slot_start)
        crm.update_item_call_time_utc(
            appt.follow_up_crm_item_id,
            reminder.isoformat(),
            field_map,
        )
    except Exception:
        logger.exception("Follow-up CRM reschedule update failed appt=%s", appt.id)


def cancel_follow_up_crm_card(
    db: Session,
    appt: Appointment,
    acting_user_id: Optional[uuid.UUID],
) -> None:
    """Mark Trello follow-up card cancelled / stop n8n pickup."""
    if not appt.follow_up_crm_item_id:
        return
    item_id = appt.follow_up_crm_item_id
    updated = False
    try:
        user_id = resolve_acting_user_id_for_follow_up(db, appt.tenant_id, acting_user_id)
        if not user_id:
            return
        crm_id = resolve_trello_crm_config_id_for_user(db, user_id)
        if not crm_id:
            return
        board_record, field_map = _trello_board_and_map(
            db, user_id=user_id, tenant_id=appt.tenant_id, crm_config_id=crm_id
        )
        if board_record.crm_type != "trello":
            return
        crm = CRMServiceFactory.get_service(CRMConfigService().get_crm_config_by_id(db, crm_id))
        if not isinstance(crm, TrelloService):
            return
        res = crm.update_item_status(
            board_record.crm_container_id,
            item_id,
            "Cancelled",
            field_map,
        )
        updated = res is not None
    except Exception:
        logger.exception("Follow-up CRM cancel failed appt=%s", appt.id)
    if updated:
        try:
            appt.follow_up_crm_item_id = None
            db.add(appt)
            db.commit()
        except Exception:
            logger.exception("Failed clearing follow_up_crm_item_id appt=%s", appt.id)
            db.rollback()
