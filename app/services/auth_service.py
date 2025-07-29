from sqlalchemy.orm import Session
from app.models.user import User
from datetime import datetime
from typing import Optional
from app.schemas.user import UserCreate
# Accepts a Pydantic schema as user_in

def create_admin_user(db: Session, user_in: UserCreate):
    new_user = User(
        email=user_in.email,
        hashed_password=user_in.hashed_password,  # hashed before calling this function
        first_name=user_in.first_name,
        last_name=user_in.last_name,
        phone=getattr(user_in, 'phone', None),
        join_date=user_in.join_date,
        created_at=getattr(user_in, 'created_at', datetime.utcnow()),
        # tenant association logic as needed
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user
