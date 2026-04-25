"""Slot reservation (in-call hold) and calendar integration."""

import os
import sys
import uuid
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.db.base import Base
from app.models.agent import Agent
from app.models.appointment import Appointment
from app.models.call_session import CallSession
from app.models.tenant import Tenant
from app.models.user import User
from app.models.slot_reservation import SlotReservation
from app.schemas.calendar import BusinessHoursUpsert
from app.services.appointment_reservation_service import appointment_reservation_service
from app.services.calendar_service import calendar_service
from app.services.call_session_contact_state import get_contact_intake, sync_contact_intake_after_message
from app.services.post_call_appointment_service import post_call_appointment_service


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(_type, compiler, **kw):
    return "JSON"


engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


TENANT_TZ = "Asia/Karachi"
# Fixed Monday so business-hours weekday matches the date regardless of "today"
MONDAY = date(2026, 6, 8)


@pytest.fixture()
def res_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    tenant = Tenant(name="ResTenant", schema_name="res_tenant")
    db.add(tenant)
    db.commit()
    u = User(
        first_name="T",
        last_name="U",
        email="t-%s@example.com" % uuid.uuid4().hex[:8],
        hashed_password="x",
        current_tenant_id=tenant.id,
    )
    db.add(u)
    db.commit()
    a = Agent(
        tenant_id=tenant.id,
        name="TestAgent",
        created_by=u.id,
        updated_by=u.id,
    )
    db.add(a)
    db.commit()
    rows = calendar_service.create_business_hours(
        db,
        tenant.id,
        [
            BusinessHoursUpsert(
                day_of_week=MONDAY.weekday(),
                open_time="09:00",
                close_time="17:00",
                timezone=TENANT_TZ,
            )
        ],
    )
    assert len(rows) == 1
    try:
        yield db, tenant, u, a
    finally:
        db.close()


def _call_session(db, tenant, u, a) -> CallSession:
    cs = CallSession(
        user_id=u.id,
        agent_id=a.id,
        tenant_id=tenant.id,
        start_time=datetime.now(timezone.utc),
        call_type="inbound",
        status="active",
    )
    db.add(cs)
    db.commit()
    return cs


def test_active_reservation_excludes_slot_from_availability(res_db):
    db, tenant, u, a = res_db
    before = calendar_service.get_available_slots(db, tenant.id, MONDAY)
    assert before.total > 0
    first = before.slots[0]

    cs = _call_session(db, tenant, u, a)
    r = appointment_reservation_service.upsert_active_reservation(
        db=db,
        tenant_id=tenant.id,
        call_session_id=cs.id,
        agent_id=a.id,
        slot_start=first.slot_start,
        metadata={"customer_name": "A", "customer_phone": "+10000000000"},
    )
    assert r.status == "active"

    after = calendar_service.get_available_slots(db, tenant.id, MONDAY)
    assert after.total == before.total - 1
    # First offered slot no longer bookable
    for s in after.slots:
        assert s.slot_start != first.slot_start


def test_upsert_replaces_same_call_hold(res_db):
    db, tenant, u, a = res_db
    slots = calendar_service.get_available_slots(db, tenant.id, MONDAY)
    assert slots.total >= 2
    a_slot = slots.slots[0].slot_start
    b_slot = slots.slots[1].slot_start
    cs = _call_session(db, tenant, u, a)
    r1 = appointment_reservation_service.upsert_active_reservation(
        db=db,
        tenant_id=tenant.id,
        call_session_id=cs.id,
        agent_id=a.id,
        slot_start=a_slot,
        metadata={},
    )
    r2 = appointment_reservation_service.upsert_active_reservation(
        db=db,
        tenant_id=tenant.id,
        call_session_id=cs.id,
        agent_id=a.id,
        slot_start=b_slot,
        metadata={},
    )
    assert r1.id != r2.id
    one = (
        db.query(SlotReservation)
        .filter(SlotReservation.call_session_id == cs.id, SlotReservation.status == "active")
        .count()
    )
    assert one == 1
    after = calendar_service.get_available_slots(db, tenant.id, MONDAY)
    # a_slot is free again
    a_times = {s.slot_start for s in after.slots}
    assert a_slot in a_times
    b_times = {s.slot_start for s in after.slots}
    assert b_slot not in b_times


def test_book_appointment_consuming_reservation_succeeds(res_db):
    db, tenant, u, a = res_db
    day_slots = calendar_service.get_available_slots(db, tenant.id, MONDAY)
    slot = day_slots.slots[0].slot_start
    cs = _call_session(db, tenant, u, a)
    r = appointment_reservation_service.upsert_active_reservation(
        db=db,
        tenant_id=tenant.id,
        call_session_id=cs.id,
        agent_id=a.id,
        slot_start=slot,
        metadata={},
    )
    with pytest.raises(ValueError, match="no longer available"):
        calendar_service.book_appointment(
            db=db,
            tenant_id=tenant.id,
            customer_name="Y",
            customer_phone="+19990002222",
            slot_start=slot,
            created_via="web",
        )

    appt = calendar_service.book_appointment(
        db=db,
        tenant_id=tenant.id,
        customer_name="X",
        customer_phone="+19990001111",
        slot_start=slot,
        call_session_id=cs.id,
        created_via="voice_agent",
        consuming_reservation_id=r.id,
    )
    assert appt.slot_start


def test_post_call_existing_appointment_releases_active_hold(res_db):
    db, tenant, u, a = res_db
    day_slots = calendar_service.get_available_slots(db, tenant.id, MONDAY)
    slot = day_slots.slots[0].slot_start
    cs = _call_session(db, tenant, u, a)

    hold = appointment_reservation_service.upsert_active_reservation(
        db=db,
        tenant_id=tenant.id,
        call_session_id=cs.id,
        agent_id=a.id,
        slot_start=slot,
        metadata={"customer_name": "Ali", "customer_phone": "+923001112233"},
    )
    assert hold.status == "active"

    # Simulate an appointment already linked to this call session.
    appt = Appointment(
        tenant_id=tenant.id,
        agent_id=a.id,
        call_session_id=cs.id,
        customer_name="Ali",
        customer_phone="+923001112233",
        slot_start=slot,
        slot_end=slot + timedelta(minutes=30),
        duration_minutes=30,
        status="pending",
        created_via="voice_agent",
    )
    db.add(appt)
    db.commit()

    post_call_appointment_service.process_call_session(db, cs.id)

    refreshed = db.query(SlotReservation).filter(SlotReservation.id == hold.id).one()
    assert refreshed.status == "released"


def test_post_call_success_with_contact_intake(res_db):
    db, tenant, u, a = res_db
    day_slots = calendar_service.get_available_slots(db, tenant.id, MONDAY)
    slot = day_slots.slots[0].slot_start
    cs = _call_session(db, tenant, u, a)
    cs.call_metadata = {
        "contact_intake": {
            "name": "John",
            "email": None,
            "name_spelled_confirmed": True,
            "email_spelled_confirmed": False,
            "name_confident": True,
            "email_validated": False,
            "name_spell_failures": 0,
            "awaiting_spell_field": None,
        },
        "booking_intent": {
            "slot_start_iso": slot.isoformat(),
            "customer_phone": "+15551234567",
            "appointment_reason": "checkup",
        },
    }
    db.add(cs)
    db.commit()

    from app.services.transcript_service import transcript_service

    transcript_service.add_message(db, cs.id, "agent", "Hello.", "speech")
    transcript_service.add_message(db, cs.id, "client", "Hi.", "speech")

    post_call_appointment_service.process_call_session(db, cs.id)

    db.refresh(cs)
    assert cs.call_metadata.get("post_call_appointment") == "success"
    appts = db.query(Appointment).filter(Appointment.call_session_id == cs.id).all()
    assert len(appts) == 1
    assert appts[0].customer_name == "John"
    assert appts[0].customer_phone == "+15551234567"


def test_post_call_fails_when_contact_not_confident(res_db):
    db, tenant, u, a = res_db
    day_slots = calendar_service.get_available_slots(db, tenant.id, MONDAY)
    slot = day_slots.slots[0].slot_start
    cs = _call_session(db, tenant, u, a)
    cs.call_metadata = {
        "contact_intake": {
            "name": None,
            "email": None,
            "name_spelled_confirmed": False,
            "email_spelled_confirmed": False,
            "name_confident": False,
            "email_validated": False,
            "name_spell_failures": 3,
            "awaiting_spell_field": None,
        },
        "booking_intent": {
            "slot_start_iso": slot.isoformat(),
            "customer_phone": "+15551234567",
            "appointment_reason": "checkup",
        },
    }
    db.add(cs)
    db.commit()

    from app.services.transcript_service import transcript_service

    transcript_service.add_message(db, cs.id, "client", "Hi.", "speech")

    post_call_appointment_service.process_call_session(db, cs.id)

    db.refresh(cs)
    assert cs.call_metadata.get("post_call_appointment") == "failed"
    assert cs.call_metadata.get("post_call_appointment_detail") == "contact_not_confident"
    assert db.query(Appointment).filter(Appointment.call_session_id == cs.id).count() == 0


def test_contact_intake_sync_after_spell_flow(res_db):
    db, tenant, u, a = res_db
    cs = _call_session(db, tenant, u, a)
    from app.services.transcript_service import transcript_service

    transcript_service.add_message(db, cs.id, "agent", "Please spell your name for me.", "speech")
    sync_contact_intake_after_message(
        db, cs.id, role="agent", message="Please spell your name for me."
    )
    transcript_service.add_message(db, cs.id, "client", "J O H N", "speech")
    sync_contact_intake_after_message(db, cs.id, role="client", message="J O H N")
    db.refresh(cs)
    intake = get_contact_intake(cs)
    assert intake["name_confident"] is True
    assert intake["name"] == "John"
    assert intake["name_spelled_confirmed"] is True
