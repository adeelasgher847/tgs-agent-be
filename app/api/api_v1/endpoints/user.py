from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.schemas.user import UserCreate, UserOut
from app.models.user import User
from app.models.tenant import Tenant
from app.api.deps import get_db
from passlib.context import CryptContext
from datetime import datetime

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@router.post("/register", response_model=UserOut)
def register_user(user_in: UserCreate, db: Session = Depends(get_db)):
    # Check if email already exists
    user = db.query(User).filter(User.email == user_in.email).first()
    if user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Validate tenant_id if provided
    if user_in.tenant_id is not None:
        tenant = db.query(Tenant).filter(Tenant.id == user_in.tenant_id).first()
        if not tenant:
            raise HTTPException(
                status_code=400, 
                detail=f"Tenant with ID {user_in.tenant_id} does not exist. Please provide a valid tenant_id or omit it to register without a tenant."
            )
    
    hashed_password = pwd_context.hash(user_in.password)
    db_user = User(
        first_name=user_in.first_name,
        last_name=user_in.last_name,
        email=user_in.email,
        phone=user_in.phone,
        hashed_password=hashed_password, 
        tenant_id=user_in.tenant_id,
        join_date=datetime.utcnow(),
        created_at=datetime.utcnow()
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user 