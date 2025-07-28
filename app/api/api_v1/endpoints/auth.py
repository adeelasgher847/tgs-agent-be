from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.schemas.user import UserCreate, UserOut
from app.models.user import User
from app.api.deps import get_db
from passlib.context import CryptContext
from datetime import datetime
from app.services.auth_service import create_admin_user

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@router.post("/register", response_model=UserOut)
def register_admin(user_in: UserCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == user_in.email).first()
    if user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = pwd_context.hash(user_in.password)
    user_in.hashed_password = hashed_password  # dynamically add hashed_password
    user_in.created_at = datetime.utcnow()     # dynamically add created_at
    db_user = create_admin_user(db, user_in)
    return db_user 