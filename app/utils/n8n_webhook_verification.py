from fastapi import Request, HTTPException, status
from app.core.config import settings
import json


def verify_n8n_webhook_secret(request: Request) -> bool:
    """
    Verify n8n webhook secret from header or body.
    
    Checks for secret in:
    1. X-N8N-Webhook-Secret header (preferred)
    2. webhook_secret in JSON body (fallback)
    
    Returns:
        bool: True if secret is valid, False otherwise
    """
    if not settings.N8N_WEBHOOK_SECRET:
        # If secret not configured, skip verification
        return False
    
    # Check header first (preferred method)
    secret_header = request.headers.get('X-N8N-Webhook-Secret')
    if secret_header and secret_header == settings.N8N_WEBHOOK_SECRET:
        return True
    
    # Check in body if it's JSON (fallback)
    try:
        # Try to get body as JSON
        if hasattr(request, '_json'):
            body = request._json
        else:
            # For FastAPI, we need to read body differently
            # This will be handled at endpoint level
            return False
        
        if isinstance(body, dict) and body.get('webhook_secret') == settings.N8N_WEBHOOK_SECRET:
            return True
    except:
        pass
    
    return False


async def verify_n8n_webhook_secret_async(request: Request) -> bool:
    """
    Async version that can read request body.
    Verify n8n webhook secret from header or body.
    """
    if not settings.N8N_WEBHOOK_SECRET:
        return False
    
    # Check header first (preferred method)
    secret_header = request.headers.get('X-N8N-Webhook-Secret')
    if secret_header and secret_header == settings.N8N_WEBHOOK_SECRET:
        return True
    
    # Check in body if it's JSON (fallback)
    try:
        body_bytes = await request.body()
        if body_bytes:
            body = json.loads(body_bytes.decode('utf-8'))
            if isinstance(body, dict) and body.get('webhook_secret') == settings.N8N_WEBHOOK_SECRET:
                return True
    except:
        pass
    
    return False

