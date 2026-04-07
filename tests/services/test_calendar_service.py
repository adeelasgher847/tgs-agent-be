from datetime import datetime, time, timedelta, timezone
import os
import sys
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.db.base import Base
from app.models.appointment import Appointment
from app.models.blocked_slot import BlockedSlot
from app.models.business_hours import BusinessHours
from app.models.tenant import Tenant
from app.schemas.calendar import BusinessHoursUpsert
from app.services.calendar_service import BusinessHoursConflictError, calendar_service


TENANT_TZ = "Asia/Karachi"


@compiles(JSONB, "sqlite")
def compile_jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _to_utc(dt_value: datetime) -> datetime:
    if dt_value.tzinfo is None:
        return dt_value.replace(tzinfo=timezone.utc)
    return dt_value.astimezone(timezone.utc)


@pytest.fixture()
def calendar_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    db = TestingSessionLocal()
    tenant = Tenant(name="Calendar Tenant", schema_name="calendar_tenant")
    db.add(tenant)
    db.commit()

    try:
        yield db
    finally:
        db.close()


def _tenant(db):
    return db.query(Tenant).first()


def _set_business_hours(db, tenant_id, target_date, *, slot_minutes=30):
    row = BusinessHours(
        tenant_id=tenant_id,
        day_of_week=target_date.weekday(),
        open_time=time(9, 0),
        close_time=time(17, 0),
        is_closed=False,
        timezone=TENANT_TZ,
        slot_duration_minutes=slot_minutes,
    )
    db.add(row)
    db.commit()
    return row


def test_create_business_hours_inserts_new_days(calendar_db):
    tenant = _tenant(calendar_db)
    payload = [
        BusinessHoursUpsert(day_of_week=0, open_time="09:00", close_time="17:00", timezone=TENANT_TZ),
    ]
    rows = calendar_service.create_business_hours(calendar_db, tenant.id, payload)
    assert len(rows) == 1
    assert rows[0].day_of_week == 0


def test_create_business_hours_rejects_duplicate_payload_days(calendar_db):
    tenant = _tenant(calendar_db)
    payload = [
        BusinessHoursUpsert(day_of_week=1, open_time="09:00", close_time="17:00", timezone=TENANT_TZ),
        BusinessHoursUpsert(day_of_week=1, open_time="10:00", close_time="18:00", timezone=TENANT_TZ),
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        calendar_service.create_business_hours(calendar_db, tenant.id, payload)


def test_create_business_hours_rejects_existing_weekday(calendar_db):
    tenant = _tenant(calendar_db)
    target_date = datetime.now(timezone.utc).date()
    _set_business_hours(calendar_db, tenant.id, target_date)
    dow = target_date.weekday()
    payload = [
        BusinessHoursUpsert(day_of_week=dow, open_time="09:00", close_time="17:00", timezone=TENANT_TZ),
    ]
    with pytest.raises(BusinessHoursConflictError) as excinfo:
        calendar_service.create_business_hours(calendar_db, tenant.id, payload)
    assert dow in excinfo.value.days


def test_delete_business_hours_removes_row(calendar_db):
    tenant = _tenant(calendar_db)
    target_date = datetime.now(timezone.utc).date()
    row = _set_business_hours(calendar_db, tenant.id, target_date)
    deleted = calendar_service.delete_business_hours(calendar_db, row.id, tenant.id)
    assert deleted is True
    stored = (
        calendar_db.query(BusinessHours)
        .filter(BusinessHours.id == row.id)
        .first()
    )
    assert stored is not None
    assert stored.is_deleted is True
    assert stored.deleted_at is not None
    assert calendar_service.get_business_hours(calendar_db, tenant.id) == []


def test_delete_business_hours_rejects_other_tenant_row(calendar_db):
    tenant = _tenant(calendar_db)
    other = Tenant(name="Other Tenant", schema_name="other_tenant")
    calendar_db.add(other)
    calendar_db.commit()
    target_date = datetime.now(timezone.utc).date()
    row = _set_business_hours(calendar_db, other.id, target_date)
    deleted = calendar_service.delete_business_hours(calendar_db, row.id, tenant.id)
    assert deleted is False
    assert (
        calendar_db.query(BusinessHours)
        .filter(BusinessHours.id == row.id)
        .first()
        is not None
    )


def test_create_business_hours_revives_soft_deleted_row(calendar_db):
    tenant = _tenant(calendar_db)
    target_date = datetime.now(timezone.utc).date()
    row = _set_business_hours(calendar_db, tenant.id, target_date)
    assert calendar_service.delete_business_hours(calendar_db, row.id, tenant.id) is True

    dow = target_date.weekday()
    payload = [
        BusinessHoursUpsert(day_of_week=dow, open_time="10:00", close_time="18:00", timezone=TENANT_TZ),
    ]
    rows = calendar_service.create_business_hours(calendar_db, tenant.id, payload)
    assert len(rows) == 1
    assert rows[0].id == row.id
    assert rows[0].is_deleted is False


def test_to_appointment_out_adds_local_fields_matching_business_timezone(calendar_db):
    tenant = _tenant(calendar_db)
    target_date = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5))).date() + timedelta(days=1)
    _set_business_hours(calendar_db, tenant.id, target_date)

    local_slot = datetime.combine(target_date, time(14, 0))
    appointment = calendar_service.book_appointment(
        db=calendar_db,
        tenant_id=tenant.id,
        customer_name="Ali",
        customer_phone="+923001112233",
        slot_start=local_slot,
        created_via="web",
    )
    out = calendar_service.to_appointment_out(calendar_db, tenant.id, appointment)
    assert out.business_timezone == TENANT_TZ
    assert out.slot_start_local is not None and out.slot_end_local is not None
    assert out.slot_start_local.hour == 14
    assert out.slot_start_local.minute == 0
    assert out.slot_start == appointment.slot_start
    assert out.slot_end == appointment.slot_end


def test_booking_uses_tenant_timezone_and_blocks_same_slot(calendar_db):
    tenant = _tenant(calendar_db)
    target_date = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5))).date() + timedelta(days=1)
    _set_business_hours(calendar_db, tenant.id, target_date)

    local_slot = datetime.combine(target_date, time(10, 0))
    appointment = calendar_service.book_appointment(
        db=calendar_db,
        tenant_id=tenant.id,
        customer_name="Ali",
        customer_phone="+923001112233",
        slot_start=local_slot,
        created_via="web",
    )

    stored_utc = _to_utc(appointment.slot_start)
    assert stored_utc.hour == 5
    assert stored_utc.minute == 0

    with pytest.raises(ValueError, match="no longer available"):
        calendar_service.book_appointment(
            db=calendar_db,
            tenant_id=tenant.id,
            customer_name="Sara",
            customer_phone="+923009998887",
            slot_start=local_slot,
            agent_id=uuid.uuid4(),
            created_via="voice_agent",
        )


def test_cancelled_slot_can_be_rebooked(calendar_db):
    tenant = _tenant(calendar_db)
    target_date = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5))).date() + timedelta(days=2)
    _set_business_hours(calendar_db, tenant.id, target_date)

    local_slot = datetime.combine(target_date, time(11, 0))
    first = calendar_service.book_appointment(
        db=calendar_db,
        tenant_id=tenant.id,
        customer_name="Ali",
        customer_phone="+923001112233",
        slot_start=local_slot,
        created_via="web",
    )

    cancelled = calendar_service.update_appointment_status(
        db=calendar_db,
        appointment_id=first.id,
        tenant_id=tenant.id,
        status="cancelled",
    )
    assert cancelled is not None
    assert cancelled.status == "cancelled"

    second = calendar_service.book_appointment(
        db=calendar_db,
        tenant_id=tenant.id,
        customer_name="Sara",
        customer_phone="+923009998887",
        slot_start=local_slot,
        created_via="voice_agent",
    )
    assert second.id != first.id


def test_off_grid_start_time_is_rejected(calendar_db):
    tenant = _tenant(calendar_db)
    target_date = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5))).date() + timedelta(days=3)
    _set_business_hours(calendar_db, tenant.id, target_date, slot_minutes=30)

    with pytest.raises(ValueError, match="slot boundaries"):
        calendar_service.book_appointment(
            db=calendar_db,
            tenant_id=tenant.id,
            customer_name="Ali",
            customer_phone="+923001112233",
            slot_start=datetime.combine(target_date, time(10, 15)),
            created_via="web",
        )


def test_availability_hides_booked_slot_for_entire_tenant(calendar_db):
    tenant = _tenant(calendar_db)
    target_date = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5))).date() + timedelta(days=4)
    _set_business_hours(calendar_db, tenant.id, target_date, slot_minutes=30)

    booked_local = datetime.combine(target_date, time(9, 30))
    calendar_service.book_appointment(
        db=calendar_db,
        tenant_id=tenant.id,
        customer_name="Ali",
        customer_phone="+923001112233",
        slot_start=booked_local,
        created_via="voice_agent",
    )

    available = calendar_service.get_available_slots(
        db=calendar_db,
        tenant_id=tenant.id,
        target_date=target_date,
        agent_id=uuid.uuid4(),
    )

    labels = [slot.slot_label for slot in available.slots]
    assert "9:30 AM" not in labels
    assert "9:00 AM" in labels


def test_reschedule_moves_to_new_slot_and_frees_old(calendar_db):
    tenant = _tenant(calendar_db)
    target_date = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5))).date() + timedelta(days=5)
    _set_business_hours(calendar_db, tenant.id, target_date, slot_minutes=30)

    slot_10 = datetime.combine(target_date, time(10, 0))
    slot_11 = datetime.combine(target_date, time(11, 0))

    first = calendar_service.book_appointment(
        db=calendar_db,
        tenant_id=tenant.id,
        customer_name="Ali",
        customer_phone="+923001112233",
        slot_start=slot_10,
        created_via="web",
    )

    orig_start = first.slot_start

    moved = calendar_service.reschedule_appointment(
        db=calendar_db,
        tenant_id=tenant.id,
        appointment_id=first.id,
        slot_start=slot_11,
    )
    assert moved.id == first.id
    assert moved.slot_start != orig_start

    # Old 10:00 slot is free for someone else
    other = calendar_service.book_appointment(
        db=calendar_db,
        tenant_id=tenant.id,
        customer_name="Sara",
        customer_phone="+923009998887",
        slot_start=slot_10,
        created_via="web",
    )
    assert other.id != first.id


def test_reschedule_rejects_overlapping_other_appointment(calendar_db):
    tenant = _tenant(calendar_db)
    target_date = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5))).date() + timedelta(days=6)
    _set_business_hours(calendar_db, tenant.id, target_date, slot_minutes=30)

    calendar_service.book_appointment(
        db=calendar_db,
        tenant_id=tenant.id,
        customer_name="Other",
        customer_phone="+923001112233",
        slot_start=datetime.combine(target_date, time(11, 0)),
        created_via="web",
    )

    mine = calendar_service.book_appointment(
        db=calendar_db,
        tenant_id=tenant.id,
        customer_name="Ali",
        customer_phone="+923009998887",
        slot_start=datetime.combine(target_date, time(10, 0)),
        created_via="web",
    )

    with pytest.raises(ValueError, match="no longer available"):
        calendar_service.reschedule_appointment(
            db=calendar_db,
            tenant_id=tenant.id,
            appointment_id=mine.id,
            slot_start=datetime.combine(target_date, time(11, 0)),
        )
