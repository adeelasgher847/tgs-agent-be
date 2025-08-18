from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session
from typing import Optional

from app.api.deps import get_db, require_tenant
from app.schemas.twilio import (
    AvailableNumbersResponse, AvailableNumberInfo,
    PhoneNumbersResponse, PhoneNumberInfo,
    PurchaseNumberRequest, UpdateNumberRequest,
    AccountInfo
)
from app.schemas.base import SuccessResponse
from app.services.twilio_service import twilio_service
from app.utils.response import create_success_response

router = APIRouter()


@router.get("/search", response_model=SuccessResponse[AvailableNumbersResponse])
async def search_available_numbers(
    country_code: str = Query("US", description="Country code (e.g., US, CA, GB)"),
    area_code: Optional[str] = Query(None, description="Specific area code"),
    contains: Optional[str] = Query(None, description="Pattern to search for in phone numbers"),
    voice_enabled: bool = Query(True, description="Whether numbers should support voice"),
    sms_enabled: bool = Query(True, description="Whether numbers should support SMS"),
    limit: int = Query(20, description="Maximum number of results"),
    user: dict = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """Search for available phone numbers"""
    try:
        numbers = twilio_service.search_available_numbers(
            country_code=country_code,
            area_code=area_code,
            contains=contains,
            voice_enabled=voice_enabled,
            sms_enabled=sms_enabled,
            limit=limit
        )
        
        number_list = [
            AvailableNumberInfo(
                phone_number=num['phone_number'],
                friendly_name=num.get('friendly_name'),
                locality=num.get('locality'),
                region=num.get('region'),
                country=num.get('country'),
                capabilities=num['capabilities'],
                beta=num['beta']
            )
            for num in numbers
        ]
        
        return create_success_response(
            AvailableNumbersResponse(numbers=number_list, total=len(number_list)),
            "Available numbers retrieved successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/purchase", response_model=SuccessResponse[PhoneNumberInfo])
async def purchase_phone_number(
    request: PurchaseNumberRequest,
    user: dict = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """Purchase a phone number"""
    try:
        # Validate phone number format
        if not twilio_service.validate_phone_number(request.phone_number):
            raise HTTPException(status_code=400, detail="Invalid phone number format. Must start with +")
        
        result = twilio_service.purchase_phone_number(
            phone_number=request.phone_number,
            webhook_url=request.webhook_url,
            status_callback_url=request.status_callback_url,
            status_callback_method=request.status_callback_method
        )
        
        return create_success_response(
            PhoneNumberInfo(
                sid=result['sid'],
                phone_number=result['phone_number'],
                friendly_name=result.get('friendly_name'),
                voice_url=result.get('voice_url'),
                voice_method=result.get('voice_method'),
                status_callback=result.get('status_callback'),
                status_callback_method=result.get('status_callback_method'),
                capabilities=result['capabilities'],
                date_created=result['date_created'],
                date_updated=result['date_updated']
            ),
            "Phone number purchased successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=SuccessResponse[PhoneNumbersResponse])
async def list_owned_numbers(
    limit: int = Query(50, description="Maximum number of results"),
    user: dict = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """List all phone numbers owned by the account"""
    try:
        numbers = twilio_service.list_owned_numbers(limit=limit)
        
        number_list = [
            PhoneNumberInfo(
                sid=num['sid'],
                phone_number=num['phone_number'],
                friendly_name=num.get('friendly_name'),
                voice_url=num.get('voice_url'),
                voice_method=num.get('voice_method'),
                status_callback=num.get('status_callback'),
                status_callback_method=num.get('status_callback_method'),
                capabilities=num['capabilities'],
                date_created=num['date_created'],
                date_updated=num['date_updated']
            )
            for num in numbers
        ]
        
        return create_success_response(
            PhoneNumbersResponse(numbers=number_list, total=len(number_list)),
            "Phone numbers retrieved successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{phone_number_sid}", response_model=SuccessResponse[PhoneNumberInfo])
async def get_number_details(
    phone_number_sid: str,
    user: dict = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """Get details of a specific phone number"""
    try:
        result = twilio_service.get_number_details(phone_number_sid)
        
        return create_success_response(
            PhoneNumberInfo(
                sid=result['sid'],
                phone_number=result['phone_number'],
                friendly_name=result.get('friendly_name'),
                voice_url=result.get('voice_url'),
                voice_method=result.get('voice_method'),
                status_callback=result.get('status_callback'),
                status_callback_method=result.get('status_callback_method'),
                capabilities=result['capabilities'],
                date_created=result['date_created'],
                date_updated=result['date_updated']
            ),
            "Phone number details retrieved successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{phone_number_sid}", response_model=SuccessResponse[PhoneNumberInfo])
async def update_number_configuration(
    phone_number_sid: str,
    request: UpdateNumberRequest,
    user: dict = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """Update configuration for a phone number"""
    try:
        result = twilio_service.update_number_configuration(
            phone_number_sid=phone_number_sid,
            friendly_name=request.friendly_name,
            webhook_url=request.webhook_url,
            status_callback_url=request.status_callback_url
        )
        
        return create_success_response(
            PhoneNumberInfo(
                sid=result['sid'],
                phone_number=result['phone_number'],
                friendly_name=result.get('friendly_name'),
                voice_url=result.get('voice_url'),
                voice_method=result.get('voice_method'),
                status_callback=result.get('status_callback'),
                status_callback_method=result.get('status_callback_method'),
                capabilities=result['capabilities'],
                date_created=result['date_created'],
                date_updated=result['date_updated']
            ),
            "Phone number configuration updated successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{phone_number_sid}")
async def release_phone_number(
    phone_number_sid: str,
    user: dict = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """Release (delete) a phone number"""
    try:
        success = twilio_service.release_phone_number(phone_number_sid)
        return create_success_response(
            {"success": success, "message": "Phone number released successfully"},
            "Phone number released successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/account/info", response_model=SuccessResponse[AccountInfo])
async def get_account_info(
    user: dict = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """Get Twilio account information"""
    try:
        result = twilio_service.get_account_info()
        
        return create_success_response(
            AccountInfo(
                sid=result['sid'],
                friendly_name=result['friendly_name'],
                status=result['status'],
                type=result['type'],
                date_created=result['date_created'],
                date_updated=result['date_updated']
            ),
            "Account information retrieved successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
