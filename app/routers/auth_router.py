from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.schemas.auth_schema import RegisterSchema
from app.services.auth_service import create_admin_user
import bcrypt

router = APIRouter(prefix="/auth", tags=["Auth"])

@router.post("/register")
def register_admin(data: RegisterSchema, db: Session = Depends(get_db)):
    hashed_password = bcrypt.hashpw(data.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    return create_admin_user(db, data.email, hashed_password, data.tenant_id)
