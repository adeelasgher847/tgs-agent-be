from fastapi import APIRouter, Request
import requests
from app.schemas.base import SuccessResponse
from app.utils.response import create_success_response

router = APIRouter()

@router.get("/health", response_model=SuccessResponse[dict])
def health_check():
    return create_success_response({"status": "ok"}, "Health check successful")

@router.get("/ip", response_model=SuccessResponse[dict])
def get_ip_address(request: Request):
    """
    Get Render server IP address for firewall whitelisting
    Returns both client IP (from request) and outbound IP (from external service)
    """
    try:
        # Get client IP from request headers (X-Forwarded-For for proxies)
        client_ip = request.client.host if request.client else None
        
        # Check for X-Forwarded-For header (common in proxy/load balancer setups)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP if multiple are present
            client_ip = forwarded_for.split(",")[0].strip()
        
        # Also check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            client_ip = real_ip
        
        # Fetch outbound IP from external service (this is what Render uses for outbound calls)
        outbound_ip = None
        try:
            response = requests.get("https://api.ipify.org?format=json", timeout=5)
            if response.status_code == 200:
                outbound_ip = response.json().get("ip")
        except Exception as e:
            pass  # Ignore errors, just return what we have
        
        return create_success_response(
            {
                "client_ip": client_ip,
                "outbound_ip": outbound_ip,
                "headers": {
                    "x-forwarded-for": request.headers.get("X-Forwarded-For"),
                    "x-real-ip": request.headers.get("X-Real-IP"),
                },
                "note": "Use 'outbound_ip' for firewall whitelisting. This is the IP Render uses for outbound API calls."
            },
            "IP addresses retrieved successfully"
        )
    except Exception as e:
        return create_success_response(
            {"error": str(e)},
            "Error retrieving IP address"
        ) 