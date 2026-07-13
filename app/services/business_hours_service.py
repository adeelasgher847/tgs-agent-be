"""
Business Hours & Blocked Slots Service
Handles CRUD for business hours configuration and blocked time slots.
All operations are scoped to tenant_id for multi-tenant isolation.
"""
from datetime import datetime, timezone
from typing import List
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
import uuid

from app.models.business_hours import BusinessHours
from app.models.blocked_slot import BlockedSlot
from app.schemas.calendar import BlockedSlotCreate, BusinessHoursUpsert


class BusinessHoursConflictError(Exception):
    """Raised when creating business hours for weekdays that already exist for the tenant."""

    def __init__(self, days: List[int]):
        self.days = days
        super().__init__()


def _parse_time_str(t: str):
    from datetime import time as dt_time
    try:
        parts = t.split(":")
        return dt_time(int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)
    except Exception:
        return None


class BusinessHoursService:
    """CRUD operations for business hours and blocked slots."""

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
        blocked_from = self._localize_and_utc(db, tenant_id, data.blocked_from)
        blocked_until = self._localize_and_utc(db, tenant_id, data.blocked_until)
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

    def _localize_and_utc(self, db: Session, tenant_id: uuid.UUID, dt_val: datetime) -> datetime:
        """Localize a naive datetime to the tenant timezone then convert to UTC."""
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
        tz_str = self.get_tenant_timezone(db, tenant_id)
        try:
            tenant_tz = ZoneInfo(tz_str)
        except (ZoneInfoNotFoundError, Exception):
            tenant_tz = timezone.utc
        if dt_val.tzinfo is None:
            dt_val = dt_val.replace(tzinfo=tenant_tz)
        else:
            dt_val = dt_val.astimezone(tenant_tz)
        if dt_val.tzinfo is None:
            return dt_val.replace(tzinfo=timezone.utc)
        return dt_val.astimezone(timezone.utc)
