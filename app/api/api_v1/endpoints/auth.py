from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.api.deps import get_db
from app.models.user import User
from app.core.security import verify_password, create_user_token
from app.schemas.auth import LoginRequest, TokenResponse

router = APIRouter()

@router.post("/login", response_model=TokenResponse)
def login(login_data: LoginRequest, db: Session = Depends(get_db)):
    """
    User login endpoint that returns JWT token.
    """
    # Find user by email
    user = db.query(User).filter(User.email == login_data.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    # Verify password
    if not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    # Get user's tenant IDs
    tenant_ids = [tenant.id for tenant in user.tenants]
    
    # Create JWT token (without tenant_ids)
    access_token = create_user_token(
        user_id=user.id,
        email=user.email,
        tenant_id=tenant_ids[0] if tenant_ids else None  # Changed from current_tenant_id
    )
    
    return TokenResponse(
        access_token=access_token,
        user_id=user.id,
        email=user.email,
        tenant_id=tenant_ids[0] if tenant_ids else None,
        tenant_ids=tenant_ids
    )



@router.post("/logout")
def logout():
    """
    Logout endpoint (client should discard token).
    Note: JWT tokens are stateless, so server-side logout requires token blacklisting.
    """
    return {"message": "Successfully logged out"} 