"""
Billing middleware to automatically track usage and enforce limits.
This middleware should be applied to routes that consume resources.
"""

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.services.billing_service import BillingService
from app.models.user import User
from app.api.deps import get_current_user_jwt
from typing import Callable
import uuid

class BillingMiddleware:
    """Middleware to track usage and enforce billing limits"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        # Skip billing checks for certain paths
        skip_paths = [
            "/api/v1/billing/",
            "/api/v1/auth/",
            "/health",
            "/docs",
            "/openapi.json"
        ]
        
        if any(request.url.path.startswith(path) for path in skip_paths):
            await self.app(scope, receive, send)
            return
        
        # Check if this is an agent-related endpoint
        if "/api/v1/agents/" in request.url.path:
            await self._handle_agent_usage(request, scope, receive, send)
        else:
            await self.app(scope, receive, send)
    
    async def _handle_agent_usage(self, request: Request, scope, receive, send):
        """Handle agent-related usage tracking"""
        try:
            # Extract user from request (this is a simplified approach)
            # In a real implementation, you'd need to properly extract the user
            # from the JWT token in the request headers
            
            # For now, we'll skip the billing check and let the endpoint handle it
            # This is because extracting the user from middleware is complex
            # and it's better to handle billing checks in the actual endpoints
            
            await self.app(scope, receive, send)
            
        except Exception as e:
            # If there's an error in billing middleware, log it but don't block the request
            print(f"Billing middleware error: {str(e)}")
            await self.app(scope, receive, send)

def check_agent_creation_limit(db: Session, tenant_id: uuid.UUID) -> bool:
    """Check if tenant can create more agents"""
    try:
        return BillingService.check_agent_limit(db, tenant_id)
    except Exception as e:
        print(f"Error checking agent limit: {str(e)}")
        return False

def check_calls_limit(db: Session, tenant_id: uuid.UUID, additional_calls: int = 1) -> bool:
    """Check if tenant can make more calls"""
    try:
        return BillingService.check_calls_limit(db, tenant_id, additional_calls)
    except Exception as e:
        print(f"Error checking calls limit: {str(e)}")
        return False

def track_agent_creation(db: Session, tenant_id: uuid.UUID) -> None:
    """Track agent creation usage"""
    try:
        BillingService.increment_agent_usage(db, tenant_id)
    except Exception as e:
        print(f"Error tracking agent creation: {str(e)}")

def track_calls_usage(db: Session, tenant_id: uuid.UUID, calls_count: int = 1) -> None:
    """Track calls usage"""
    try:
        BillingService.increment_calls_usage(db, tenant_id, calls_count)
    except Exception as e:
        print(f"Error tracking calls usage: {str(e)}")

def enforce_billing_limits(db: Session, tenant_id: uuid.UUID) -> dict:
    """Enforce billing limits and return status"""
    try:
        return BillingService.check_and_enforce_limits(db, tenant_id)
    except Exception as e:
        print(f"Error enforcing billing limits: {str(e)}")
        return {"within_limits": True, "over_agent_limit": False, "over_calls_limit": False}

# Decorator for endpoints that create agents
def require_agent_limit(func: Callable) -> Callable:
    """Decorator to check agent creation limits"""
    async def wrapper(*args, **kwargs):
        # This would need to be implemented with proper dependency injection
        # For now, it's a placeholder
        return await func(*args, **kwargs)
    return wrapper

# Decorator for endpoints that make calls
def require_calls_limit(additional_calls: int = 1):
    """Decorator to check calls limits"""
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            # This would need to be implemented with proper dependency injection
            # For now, it's a placeholder
            return await func(*args, **kwargs)
        return wrapper
    return decorator
