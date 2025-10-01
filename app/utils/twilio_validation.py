import hmac
import hashlib
import base64
from fastapi import Request, HTTPException
from app.core.config import settings


def validate_twilio_signature(request: Request, body: str) -> bool:
    """Validate Twilio webhook signature"""
    try:
        # Get the signature from headers
        signature = request.headers.get('X-Twilio-Signature')
        if not signature:
            return False
        
        # Get the full URL
        url = str(request.url)
        
        # Get auth token from settings
        auth_token = settings.TWILIO_AUTH_TOKEN
        if not auth_token:
            return False
        
        # Create expected signature
        expected_signature = base64.b64encode(
            hmac.new(
                auth_token.encode('utf-8'),
                (url + body).encode('utf-8'),
                hashlib.sha1
            ).digest()
        ).decode('utf-8')
        
        return hmac.compare_digest(signature, expected_signature)
    
    except Exception as e:
        print(f"Error validating Twilio signature: {e}")
        return False


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
