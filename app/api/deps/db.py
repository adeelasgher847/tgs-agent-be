from typing import Generator, Optional
import uuid

from sqlalchemy.exc import InterfaceError
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.db.async_session import get_db as get_async_db  # noqa: F401 — re-exported for callers
from app.models.user import User


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        try:
            db.close()
        except InterfaceError:
            # During app shutdown/reload, connection can already be gone.
            pass


def get_active_user_by_id(db: Session, user_id: uuid.UUID) -> Optional[User]:
    """Load a user only when not soft-deleted (``deleted_at IS NULL``)."""
    return (
        db.query(User)
        .filter(User.id == user_id, User.deleted_at.is_(None))
        .first()
    )
