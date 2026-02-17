"""
Carrier Router
API endpoints for carrier management
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import uuid
import requests
import urllib3

from app.api.deps import get_db, require_tenant
from app.models.user import User
from app.schemas.carrier import CarrierCreate, CarrierUpdate, CarrierResponse, CarrierList
from app.schemas.base import SuccessResponse
from app.services.carrier_service import carrier_service
from app.utils.response import create_success_response
from app.core.logger import logger

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

router = APIRouter()


@router.get("/", response_model=SuccessResponse[CarrierList])
async def get_carriers(
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """Get all carriers (global + tenant-specific)"""
    try:
        # Get global carriers + tenant-specific carriers
        carriers = carrier_service.get_carriers(db, user.current_tenant_id)
        
        carrier_responses = [
            CarrierResponse(
                id=carrier.id,
                tenant_id=carrier.tenant_id,
                name=carrier.name,
                provider=carrier.provider,
                status=carrier.status,
                description=carrier.description,
                sip_server=carrier.sip_server,
                sip_port=carrier.sip_port,
                vicidial_carrier_id=carrier.vicidial_carrier_id,
                created_at=carrier.created_at,
                updated_at=carrier.updated_at
            )
            for carrier in carriers
        ]
        
        return create_success_response(
            CarrierList(
                carriers=carrier_responses,
                total=len(carrier_responses)
            ),
            "Carriers retrieved successfully"
        )
    except Exception as e:
        logger.error(f"❌ Error getting carriers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test-vicidial-ip-validation")
async def test_vicidial_ip_validation(
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Test Vicidial IP validation by hitting teleconnect.php endpoint
    This validates that Render server IP can access Vicidial server
    
    Endpoint: http://vagenttgs.com:81/teleconnect.php
    User: 1001
    Pass: 1001
    """
    try:
        # Vicidial teleconnect.php URL
        teleconnect_url = "http://vagenttgs.com:81/teleconnect.php"
        
        # Test credentials (as specified by user)
        test_user = "1001"
        test_pass = "1001"
        
        # Prepare POST data for teleconnect.php
        # Note: teleconnect.php typically expects POST with phone_login, phone_pass, user_login, user_pass
        post_data = {
            "phone_login": test_user,
            "phone_pass": test_pass,
            "user_login": test_user,
            "user_pass": test_pass
        }
        
        logger.info(f"🔍 Testing Vicidial IP validation: {teleconnect_url} with user={test_user}")
        
        # Make request to teleconnect.php
        try:
            response = requests.post(
                teleconnect_url,
                data=post_data,
                timeout=10,
                allow_redirects=False  # Don't follow redirects, we just want to check if connection works
            )
            
            # Check response
            status_code = response.status_code
            response_text = response.text[:500] if response.text else ""  # First 500 chars
            
            # Success criteria:
            # - Status 200: Connection successful, page loaded
            # - Status 302/301: Redirect (likely means login successful or redirect to agent screen)
            # - Status 403: Forbidden (IP blocked)
            # - Connection refused/timeout: IP not whitelisted or server down
            
            if status_code == 200:
                logger.info(f"✅ Vicidial IP validation successful: Status {status_code}")
                return {
                    "success": True,
                    "message": "Vicidial server is accessible from Render IP",
                    "status_code": status_code,
                    "response_preview": response_text,
                    "render_ip": "Request sent from Render server IP",
                    "test_url": teleconnect_url,
                    "test_user": test_user
                }
            elif status_code in [301, 302, 303, 307, 308]:
                # Redirect usually means login successful or page redirect
                redirect_location = response.headers.get("Location", "")
                logger.info(f"✅ Vicidial IP validation successful: Redirect {status_code} to {redirect_location}")
                return {
                    "success": True,
                    "message": "Vicidial server is accessible from Render IP (redirect received)",
                    "status_code": status_code,
                    "redirect_location": redirect_location,
                    "render_ip": "Request sent from Render server IP",
                    "test_url": teleconnect_url,
                    "test_user": test_user
                }
            elif status_code == 403:
                logger.warning(f"⚠️ Vicidial IP validation failed: Forbidden (Status {status_code})")
                return {
                    "success": False,
                    "message": "Vicidial server returned Forbidden - IP may be blocked",
                    "status_code": status_code,
                    "response_preview": response_text,
                    "render_ip": "Request sent from Render server IP",
                    "test_url": teleconnect_url,
                    "test_user": test_user,
                    "action_required": "Check Vicidial server firewall/whitelist settings"
                }
            else:
                logger.warning(f"⚠️ Vicidial IP validation returned unexpected status: {status_code}")
                return {
                    "success": False,
                    "message": f"Vicidial server returned status {status_code}",
                    "status_code": status_code,
                    "response_preview": response_text,
                    "render_ip": "Request sent from Render server IP",
                    "test_url": teleconnect_url,
                    "test_user": test_user
                }
                
        except requests.exceptions.ConnectionError as e:
            error_msg = str(e)
            logger.error(f"❌ Vicidial IP validation failed: Connection refused - {error_msg}")
            return {
                "success": False,
                "message": "Connection refused - Vicidial server not accessible from Render IP",
                "error": error_msg,
                "render_ip": "Request sent from Render server IP",
                "test_url": teleconnect_url,
                "test_user": test_user,
                "action_required": "Check if port 81 is open and Render IP is whitelisted on Vicidial server"
            }
        except requests.exceptions.Timeout as e:
            logger.error(f"❌ Vicidial IP validation failed: Timeout - {e}")
            return {
                "success": False,
                "message": "Request timeout - Vicidial server not responding",
                "error": str(e),
                "render_ip": "Request sent from Render server IP",
                "test_url": teleconnect_url,
                "test_user": test_user,
                "action_required": "Check Vicidial server status and network connectivity"
            }
        except Exception as e:
            logger.error(f"❌ Vicidial IP validation failed: {e}")
            return {
                "success": False,
                "message": "Error testing Vicidial connection",
                "error": str(e),
                "render_ip": "Request sent from Render server IP",
                "test_url": teleconnect_url,
                "test_user": test_user
            }
            
    except Exception as e:
        logger.error(f"❌ Error in test_vicidial_ip_validation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/{carrier_id}", response_model=SuccessResponse[CarrierResponse])
async def get_carrier(
    carrier_id: uuid.UUID,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """Get a specific carrier by ID"""
    try:
        carrier = carrier_service.get_carrier_by_id(db, carrier_id, user.current_tenant_id)
        
        if not carrier:
            raise HTTPException(status_code=404, detail="Carrier not found")
        
        return create_success_response(
            CarrierResponse(
                id=carrier.id,
                tenant_id=carrier.tenant_id,
                name=carrier.name,
                provider=carrier.provider,
                status=carrier.status,
                description=carrier.description,
                sip_server=carrier.sip_server,
                sip_port=carrier.sip_port,
                vicidial_carrier_id=carrier.vicidial_carrier_id,
                created_at=carrier.created_at,
                updated_at=carrier.updated_at
            ),
            "Carrier retrieved successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting carrier: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=SuccessResponse[CarrierResponse])
async def create_carrier(
    request: CarrierCreate,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """Create a new carrier (global if tenant_id not provided, tenant-specific if provided)
    
    Note: tenant_id is optional. If not provided, carrier will be global (available to all tenants).
    """
    try:
        # tenant_id is optional - if not provided or None, carrier will be global
        # Use request directly - tenant_id is optional in schema
        carrier_data = CarrierCreate(**request.dict(exclude_unset=True))
        
        carrier = carrier_service.create_carrier(db, carrier_data)
        
        return create_success_response(
            CarrierResponse(
                id=carrier.id,
                tenant_id=carrier.tenant_id,
                name=carrier.name,
                provider=carrier.provider,
                status=carrier.status,
                description=carrier.description,
                sip_server=carrier.sip_server,
                sip_port=carrier.sip_port,
                vicidial_carrier_id=carrier.vicidial_carrier_id,
                created_at=carrier.created_at,
                updated_at=carrier.updated_at
            ),
            "Carrier created successfully"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Error creating carrier: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{carrier_id}", response_model=SuccessResponse[CarrierResponse])
async def update_carrier(
    carrier_id: uuid.UUID,
    request: CarrierUpdate,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """Update a carrier"""
    try:
        carrier = carrier_service.update_carrier(db, carrier_id, user.current_tenant_id, request)
        
        if not carrier:
            raise HTTPException(status_code=404, detail="Carrier not found")
        
        return create_success_response(
            CarrierResponse(
                id=carrier.id,
                tenant_id=carrier.tenant_id,
                name=carrier.name,
                provider=carrier.provider,
                status=carrier.status,
                description=carrier.description,
                sip_server=carrier.sip_server,
                sip_port=carrier.sip_port,
                vicidial_carrier_id=carrier.vicidial_carrier_id,
                created_at=carrier.created_at,
                updated_at=carrier.updated_at
            ),
            "Carrier updated successfully"
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Error updating carrier: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{carrier_id}", response_model=SuccessResponse[dict])
async def delete_carrier(
    carrier_id: uuid.UUID,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """Delete a carrier"""
    try:
        success = carrier_service.delete_carrier(db, carrier_id, user.current_tenant_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Carrier not found")
        
        return create_success_response(
            {"deleted": True},
            "Carrier deleted successfully"
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Error deleting carrier: {e}")
        raise HTTPException(status_code=500, detail=str(e))
