from datetime import datetime, timedelta, timezone
from typing import Optional, List
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.core.config import settings
import uuid
import secrets

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token with expiration"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        # Use the configured expiration time (15 minutes by default)
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[dict]:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        return None

def is_token_expired(token: str) -> bool:
    """Check if token is expired"""
    payload = verify_token(token)
    if not payload:
        return True
    
    exp_timestamp = payload.get("exp")
    if not exp_timestamp:
        return True
    
    exp_time = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
    return datetime.now(timezone.utc) > exp_time

def get_token_info(token: str) -> Optional[dict]:
    """Get comprehensive token information"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        exp_timestamp = payload.get("exp")
        iat_timestamp = payload.get("iat")
        
        token_info = {
            "user_id": payload.get("user_id"),
            "email": payload.get("email"),
            "tenant_id": payload.get("tenant_id"),
            "type": payload.get("type"),
            "expires_at": datetime.fromtimestamp(exp_timestamp, tz=timezone.utc).isoformat() if exp_timestamp else None,
            "issued_at": datetime.fromtimestamp(iat_timestamp, tz=timezone.utc).isoformat() if iat_timestamp else None,
            "is_expired": False,
            "expires_in_minutes": None
        }
        
        if exp_timestamp:
            exp_time = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
            now = datetime.now(timezone.utc)
            token_info["is_expired"] = now > exp_time
            if not token_info["is_expired"]:
                remaining = exp_time - now
                token_info["expires_in_minutes"] = int(remaining.total_seconds() / 60)
        
        return token_info
    except JWTError:
        return None

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash password"""
    return pwd_context.hash(password)

def create_user_token(user_id: uuid.UUID, email: str, tenant_id: Optional[uuid.UUID] = None):
    """
    Create JWT token for user with 15-minute expiration
    
    Args:
        user_id: User's ID (UUID)
        email: User's email
        tenant_id: Current tenant ID (UUID, optional)
    
    Returns:
        JWT token that expires in 15 minutes
    """
    token_data = {
        "user_id": str(user_id),  # Convert UUID to string
        "email": email,
        "tenant_id": str(tenant_id) if tenant_id else None,  # Convert UUID to string
        "iat": datetime.now(timezone.utc),  # Issued at
        "type": "access"
    }
    
    # Explicitly set 15-minute expiration
    expires_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return create_access_token(data=token_data, expires_delta=expires_delta)

def generate_password_reset_token() -> str:
    """
    Generate a secure random token for password reset
    
    Returns:
        str: A secure random token
    """
    return secrets.token_urlsafe(32)

def create_password_reset_token(user_id: uuid.UUID) -> tuple[str, datetime]:
    """
    Create a password reset token with expiration
    
    Args:
        user_id: User's ID (UUID)
    
    Returns:
        tuple: (token, expiration_datetime)
    """
    token = generate_password_reset_token()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.PASSWORD_RESET_TOKEN_EXPIRE_MINUTES)
    return token, expires_at

def create_refresh_token_value() -> str:
    """Create secure random refresh token string."""
    return secrets.token_urlsafe(48)

def refresh_token_expires_at() -> datetime:
    """Get refresh token expiration time (from minutes setting)."""
    return datetime.now(timezone.utc) + timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)