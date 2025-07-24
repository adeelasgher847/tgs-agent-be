from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.schemas.user import UserCreate, UserOut
from app.models.user import User
from app.models.role import Role
from app.api.deps import get_db
from passlib.context import CryptContext

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@router.post("/register", response_model=UserOut)
def register_user(user_in: UserCreate, db: Session = Depends(get_db)):
    # Check if email already exists
    user = db.query(User).filter(User.email == user_in.email).first()
    if user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Set default role_id to "user" (ID: 2) if not provided
    role_id = user_in.role_id if user_in.role_id is not None else 2
    
    # Validate role_id
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(
            status_code=400, 
            detail=f"Role with ID {role_id} does not exist. Please provide a valid role_id or omit it to use default 'user' role."
        )
    
    hashed_password = pwd_context.hash(user_in.password)
    db_user = User(
        email=user_in.email,
        hashed_password=hashed_password,
        role_id=role_id
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user 