from app.db.session import SessionLocal
from typing import Generator
from fastapi import Depends, HTTPException, Header
from sqlalchemy.orm import Session
from app.models.user import User

def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(
    x_user_id: int = Header(..., description="User ID for authentication"),
    db: Session = Depends(get_db)
) -> User:
    """
    Simple user authentication via header.
    In production, this would be replaced with JWT token validation.
    """
    user = db.query(User).filter(User.id == x_user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user 