# from fastapi import APIRouter, HTTPException, status, Depends
# from sqlalchemy.orm import Session
# from app.db.session import get_db
# from app.schemas.auth_schema import RegisterSchema
# from app.services.auth_service import create_admin_user
# from app.models.user import User
# import bcrypt

# router = APIRouter(prefix="/auth", tags=["Auth"])


# @router.post("/register")
# def register_admin(data: RegisterSchema, db: Session = Depends(get_db)):
#     normalized_email = data.email.strip().lower()

    
#     existing_user = db.query(User).filter(User.email == normalized_email).first()
#     if existing_user:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail=f"Email already exists: {data.email}"
#         )

   
#     hashed_password = bcrypt.hashpw(data.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    
#     return create_admin_user(db, normalized_email, hashed_password, data.tenant_id)
