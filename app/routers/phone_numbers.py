from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid

from app.api.deps import get_db, require_tenant
from app.models.user import User
from app.schemas.phone_number import (
    PhoneNumberResponse, PhoneNumberList,
    CreatePhoneNumberRequest, CreatePhoneNumberResponse,
    PhoneNumberUpdate,
    ImportTwilioPhoneNumberRequest, ImportTwilioPhoneNumberResponse
)
from app.schemas.base import SuccessResponse
from app.services.phone_number_service import phone_number_service
from app.services.twilio_service import twilio_service
from app.utils.response import create_success_response
from app.core.logger import logger

router = APIRouter()

@router.get("/", response_model=SuccessResponse[PhoneNumberList])
async def get_phone_numbers(
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """Get all phone numbers for the tenant"""
    try:
        phone_numbers = phone_number_service.get_phone_numbers(db, user.current_tenant_id)
        
        phone_number_responses = []
        for pn in phone_numbers:
            # Don't return encrypted credentials in response (security)
            account_sid_display = "***encrypted***" if pn.twilio_account_sid else None
            phone_number_responses.append(PhoneNumberResponse(
                id=pn.id,
                tenant_id=pn.tenant_id,
                phone_number=pn.phone_number,
                label=pn.label,
                status=pn.status,
                assistant_id=pn.assistant_id,
                twilio_phone_number_sid=pn.twilio_phone_number_sid,
                twilio_account_sid=account_sid_display,  # ✅ Don't expose encrypted value
                created_at=pn.created_at,
                updated_at=pn.updated_at
            ))
        
        return create_success_response(
            PhoneNumberList(
                phone_numbers=phone_number_responses,
                total=len(phone_number_responses)
            ),
            f"Retrieved {len(phone_number_responses)} phone numbers"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", response_model=SuccessResponse[CreatePhoneNumberResponse])
async def create_phone_number(
    request: CreatePhoneNumberRequest,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """Create a new phone number"""
    try:
        from app.schemas.phone_number import PhoneNumberCreate
        from app.models.agent import Agent
        
        # Validate assistant_id if provided
        if request.assistant_id:
            agent = db.query(Agent).filter(
                Agent.id == request.assistant_id,
                Agent.tenant_id == user.current_tenant_id
            ).first()
            if not agent:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Assistant with ID {request.assistant_id} not found or doesn't belong to your tenant"
                )
        
        phone_number_data = PhoneNumberCreate(
            phone_number=request.phone_number,
            label=request.label,
            assistant_id=request.assistant_id,  # This can be None
            tenant_id=user.current_tenant_id
        )
        
        phone_number = phone_number_service.create_phone_number(db, phone_number_data)
        
        return create_success_response(
            CreatePhoneNumberResponse(
                id=phone_number.id,
                phone_number=phone_number.phone_number,
                label=phone_number.label,
                status=phone_number.status,
                created_at=phone_number.created_at,
                message="Phone number created successfully"
            ),
            "Phone number created successfully"
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/import", response_model=SuccessResponse[ImportTwilioPhoneNumberResponse])
async def import_twilio_phone_number(
    request: ImportTwilioPhoneNumberRequest,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Import a Twilio phone number with custom Account SID and Auth Token.
    User can use their own Twilio account credentials.
    
    This allows users to:
    - Use their own Twilio account for calls
    - Keep credentials secure (encrypted in database)
    - Assign the number to an agent for automatic use
    """
    try:
        # Import phone number
        phone_number = phone_number_service.import_twilio_phone_number(
            db=db,
            phone_number=request.phone_number,
            label=request.label,
            tenant_id=user.current_tenant_id,
            twilio_account_sid=request.twilio_account_sid,
            twilio_auth_token=request.twilio_auth_token
        )
        
        return create_success_response(
            ImportTwilioPhoneNumberResponse(
                id=phone_number.id,
                phone_number=phone_number.phone_number,
                label=phone_number.label,
                status=phone_number.status,
                twilio_account_sid="***encrypted***",  # ✅ Don't expose encrypted value
                created_at=phone_number.created_at,
                message="Twilio phone number imported successfully"
            ),
            "Twilio phone number imported successfully"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import phone number: {str(e)}")

# Specific routes must come before parameterized routes
@router.get("/available-numbers", include_in_schema=False)
async def get_available_phone_numbers(
    country_code: str = Query(default="US", description="Country code (e.g., US, CA, GB)"),
    area_code: Optional[str] = Query(default=None, description="Specific area code to search for"),
    contains: Optional[str] = Query(default=None, description="Pattern to search for in phone numbers"),
    voice_enabled: bool = Query(default=True, description="Whether numbers should support voice"),
    sms_enabled: bool = Query(default=True, description="Whether numbers should support SMS"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of results to return"),
    user: User = Depends(require_tenant)
):
    """Get available phone numbers from Twilio"""
    try:
        logger.info(f"🔍 Searching for available phone numbers")
        logger.debug(f"📞 Country: {country_code}, Area Code: {area_code}")
        logger.debug(f"🔍 Contains: {contains}, Voice: {voice_enabled}, SMS: {sms_enabled}")
        
        available_numbers = twilio_service.search_available_numbers(
            country_code=country_code,
            area_code=area_code,
            contains=contains,
            voice_enabled=voice_enabled,
            sms_enabled=sms_enabled,
            limit=limit
        )
        
        logger.info(f"✅ Found {len(available_numbers)} available numbers")
        
        return create_success_response(
            {
                "available_numbers": available_numbers,
                "total": len(available_numbers),
                "search_params": {
                    "country_code": country_code,
                    "area_code": area_code,
                    "contains": contains,
                    "voice_enabled": voice_enabled,
                    "sms_enabled": sms_enabled,
                    "limit": limit
                }
            },
            f"Found {len(available_numbers)} available phone numbers"
        )
        
    except Exception as e:
        logger.error(f"❌ Error searching for available numbers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to search for available numbers: {str(e)}")

@router.get("/available-number")
async def get_owned_phone_numbers(
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of results to return"),
    user: User = Depends(require_tenant)
):
    """Get all phone numbers owned by your Twilio account"""
    try:
        logger.info(f"📱 Fetching owned phone numbers from Twilio")
        
        owned_numbers = twilio_service.list_owned_numbers(limit=limit)
        
        logger.info(f"✅ Found {len(owned_numbers)} owned numbers")
        
        return create_success_response(
            {
                "owned_numbers": owned_numbers,
                "total": len(owned_numbers)
            },
            f"Found {len(owned_numbers)} owned phone numbers"
        )
        
    except Exception as e:
        logger.error(f"❌ Error fetching owned numbers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch owned numbers: {str(e)}")

@router.get("/twilio/account-info", include_in_schema=False)
async def get_twilio_account_info(
    user: User = Depends(require_tenant)
):
    """Get Twilio account information"""
    try:
        logger.info(f"🏦 Fetching Twilio account information")
        
        account_info = twilio_service.get_account_info()
        
        logger.info(f"✅ Account info retrieved: {account_info.get('friendly_name', 'Unknown')}")
        
        return create_success_response(
            {
                "account_info": account_info
            },
            "Twilio account information retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"❌ Error fetching account info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch account info: {str(e)}")

@router.post("/twilio/purchase", include_in_schema=False)
async def purchase_phone_number(
    phone_number: str = Query(..., description="Phone number to purchase (e.g., +1234567890)"),
    webhook_url: Optional[str] = Query(default=None, description="Webhook URL for incoming calls"),
    status_callback_url: Optional[str] = Query(default=None, description="Webhook URL for call status updates"),
    user: User = Depends(require_tenant)
):
    """Purchase a phone number from Twilio"""
    try:
        logger.info(f"💰 Purchasing phone number: {phone_number}")
        
        # Build webhook URLs if not provided
        if not webhook_url:
            from app.core.config import settings
            webhook_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/incoming"
        
        if not status_callback_url:
            from app.core.config import settings
            status_callback_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/call-events"
        
        purchase_result = twilio_service.purchase_phone_number(
            phone_number=phone_number,
            webhook_url=webhook_url,
            status_callback_url=status_callback_url
        )
        
        logger.info(f"✅ Phone number purchased successfully: {purchase_result['phone_number']}")
        
        return create_success_response(
            {
                "purchase_result": purchase_result
            },
            f"Phone number {phone_number} purchased successfully"
        )
        
    except Exception as e:
        logger.error(f"❌ Error purchasing phone number: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to purchase phone number: {str(e)}")

@router.get("/twilio/verified", include_in_schema=False)
async def get_verified_phone_numbers(
    user: User = Depends(require_tenant)
):
    """Get verified phone numbers from Twilio (for outbound calls)"""
    try:
        logger.info(f"✅ Fetching verified phone numbers from Twilio")
        
        # Get owned numbers (these are verified for outbound calls)
        owned_numbers = twilio_service.list_owned_numbers(limit=100)
        
        # Filter for numbers that can make outbound calls
        verified_numbers = []
        for number in owned_numbers:
            if number.get('capabilities', {}).get('voice', False):
                verified_numbers.append({
                    'phone_number': number['phone_number'],
                    'friendly_name': number.get('friendly_name', ''),
                    'sid': number['sid'],
                    'capabilities': number['capabilities'],
                    'date_created': number['date_created']
                })
        
        logger.info(f"✅ Found {len(verified_numbers)} verified phone numbers")
        
        return create_success_response(
            {
                "verified_numbers": verified_numbers,
                "total": len(verified_numbers)
            },
            f"Found {len(verified_numbers)} verified phone numbers for outbound calls"
        )
        
    except Exception as e:
        logger.error(f"❌ Error fetching verified numbers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch verified numbers: {str(e)}")

@router.get("/{phone_number_id}", response_model=SuccessResponse[PhoneNumberResponse])
async def get_phone_number(
    phone_number_id: uuid.UUID,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """Get a specific phone number"""
    try:
        phone_number = phone_number_service.get_phone_number_by_id(db, phone_number_id, user.current_tenant_id)
        
        if not phone_number:
            raise HTTPException(status_code=404, detail="Phone number not found")
        
        return create_success_response(
            PhoneNumberResponse(
                id=phone_number.id,
                tenant_id=phone_number.tenant_id,
                phone_number=phone_number.phone_number,
                label=phone_number.label,
                status=phone_number.status,
                assistant_id=phone_number.assistant_id,
                twilio_phone_number_sid=phone_number.twilio_phone_number_sid,
                created_at=phone_number.created_at,
                updated_at=phone_number.updated_at
            ),
            "Phone number retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{phone_number_id}", response_model=SuccessResponse[PhoneNumberResponse])
async def update_phone_number(
    phone_number_id: uuid.UUID,
    request: PhoneNumberUpdate,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """Update a phone number"""
    try:
        phone_number = phone_number_service.update_phone_number(db, phone_number_id, user.current_tenant_id, request)
        
        if not phone_number:
            raise HTTPException(status_code=404, detail="Phone number not found")
        
        return create_success_response(
            PhoneNumberResponse(
                id=phone_number.id,
                tenant_id=phone_number.tenant_id,
                phone_number=phone_number.phone_number,
                label=phone_number.label,
                status=phone_number.status,
                assistant_id=phone_number.assistant_id,
                twilio_phone_number_sid=phone_number.twilio_phone_number_sid,
                created_at=phone_number.created_at,
                updated_at=phone_number.updated_at
            ),
            "Phone number updated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{phone_number_id}", response_model=SuccessResponse[dict])
async def delete_phone_number(
    phone_number_id: uuid.UUID,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """Delete a phone number"""
    try:
        success = phone_number_service.delete_phone_number(db, phone_number_id, user.current_tenant_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Phone number not found")
        
        return create_success_response(
            {"message": "Phone number deleted successfully"},
            "Phone number deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
