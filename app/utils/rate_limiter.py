from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from fastapi import Request, HTTPException, status
import redis.asyncio as redis
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Global limiter instance
limiter = None

async def init_rate_limiter():
    """Initialize the rate limiter with Redis connection."""
    global limiter
    try:
        redis_client = redis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
        limiter = FastAPILimiter(redis_client)
        logger.info("Rate limiter initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize rate limiter: {e}")
        # If Redis is not available, disable rate limiting
        limiter = None

async def close_rate_limiter():
    """Close the rate limiter connection."""
    global limiter
    if limiter:
        await limiter.close()
        limiter = None

def get_rate_limiter():
    """Get the rate limiter instance."""
    return limiter

# Rate limiting decorators
def login_rate_limit():
    """Rate limiter for login endpoints."""
    if not settings.RATE_LIMIT_ENABLED or not limiter:
        return lambda func: func
    
    return RateLimiter(
        times=settings.LOGIN_RATE_LIMIT,
        seconds=settings.LOGIN_RATE_WINDOW,
        key_func=lambda request: f"login:{request.client.host if request.client else 'unknown'}"
    )

def webhook_rate_limit():
    """Rate limiter for webhook endpoints."""
    if not settings.RATE_LIMIT_ENABLED or not limiter:
        return lambda func: func
    
    return RateLimiter(
        times=settings.WEBHOOK_RATE_LIMIT,
        seconds=settings.WEBHOOK_RATE_WINDOW,
        key_func=lambda request: f"webhook:{request.client.host if request.client else 'unknown'}"
    )

def api_rate_limit():
    """Rate limiter for general API endpoints."""
    if not settings.RATE_LIMIT_ENABLED or not limiter:
        return lambda func: func
    
    return RateLimiter(
        times=settings.API_RATE_LIMIT,
        seconds=settings.API_RATE_WINDOW,
        key_func=lambda request: f"api:{request.client.host if request.client else 'unknown'}"
    )

def custom_rate_limit(times: int, seconds: int, key_prefix: str = "custom"):
    """Create a custom rate limiter."""
    if not settings.RATE_LIMIT_ENABLED or not limiter:
        return lambda func: func
    
    return RateLimiter(
        times=times,
        seconds=seconds,
        key_func=lambda request: f"{key_prefix}:{request.client.host if request.client else 'unknown'}"
    )

# Rate limit error handler
def rate_limit_exceeded_handler(request: Request, exc: Exception):
    """Handle rate limit exceeded errors."""
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="Rate limit exceeded. Please try again later.",
        headers={"Retry-After": str(settings.LOGIN_RATE_WINDOW)}
    )
