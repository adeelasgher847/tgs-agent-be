from fastapi import Request
from twilio.request_validator import RequestValidator
from app.core.config import settings
from app.core.logger import logger


def _first_header_value(value: str | None) -> str | None:
    if not value:
        return None
    return value.split(",")[0].strip()


def build_twilio_validation_url(request: Request) -> str:
    """
    Build a proxy-aware URL for Twilio signature validation.
    Twilio signs the publicly reachable URL, which can differ behind proxies.
    """
    forwarded_proto = _first_header_value(request.headers.get("X-Forwarded-Proto"))
    forwarded_host = _first_header_value(request.headers.get("X-Forwarded-Host"))

    scheme = forwarded_proto or request.url.scheme
    host = forwarded_host or request.headers.get("host") or request.url.netloc

    path = request.url.path
    query = request.url.query
    if query:
        return f"{scheme}://{host}{path}?{query}"
    return f"{scheme}://{host}{path}"


def validate_twilio_signature_with_token(
    request: Request, params: dict, auth_token: str
) -> bool:
    """
    Validate Twilio webhook signature with an explicit auth token.
    `params` must be a dict of the parsed form fields (not a raw body string).
    Twilio's RequestValidator signs form params as a sorted key-value dict.
    """
    try:
        signature = request.headers.get("X-Twilio-Signature")
        if not signature:
            return False

        if not auth_token:
            return False

        url = build_twilio_validation_url(request)
        validator = RequestValidator(auth_token)
        return validator.validate(url, params, signature)

    except Exception as e:
        logger.error(f"Error validating Twilio signature: {e}")
        return False


def validate_twilio_signature(request: Request, params: dict) -> bool:
    """Validate Twilio webhook signature using global settings token."""
    return validate_twilio_signature_with_token(request, params, settings.TWILIO_AUTH_TOKEN)


def validate_webrtc_auth(request: Request) -> bool:
    """Validate WebRTC authentication (placeholder implementation)"""
    # This is a placeholder - implement your WebRTC auth logic here
    auth_token = request.headers.get('Authorization')
    if not auth_token:
        return False
    
    # Add your WebRTC authentication logic here
    # For now, we'll accept any Authorization header
    return auth_token.startswith('Bearer ')


async def get_request_body(request: Request) -> str:
    """Get request body as string"""
    body = await request.body()
    return body.decode('utf-8')
