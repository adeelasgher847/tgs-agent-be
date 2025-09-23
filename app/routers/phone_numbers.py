from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List
import uuid

from app.api.deps import get_db, require_tenant
from app.models.user import User
from app.schemas.phone_number import (
    PhoneNumberResponse, PhoneNumberList,
    CreatePhoneNumberRequest, CreatePhoneNumberResponse,
    PhoneNumberUpdate
)
from app.services.phone_number_service import phone_number_service
from app.utils.response import create_success_response

router = APIRouter()

@router.get("/", response_model=PhoneNumberList)
async def get_phone_numbers(
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """Get all phone numbers for the tenant"""
    try:
        phone_numbers = phone_number_service.get_phone_numbers(db, user.current_tenant_id)
        
        phone_number_responses = []
        for pn in phone_numbers:
            phone_number_responses.append(PhoneNumberResponse(
                id=pn.id,
                tenant_id=pn.tenant_id,
                phone_number=pn.phone_number,
                label=pn.label,
                status=pn.status,
                assistant_id=pn.assistant_id,
                twilio_phone_number_sid=pn.twilio_phone_number_sid,
                created_at=pn.created_at,
                updated_at=pn.updated_at
            ))
        
        return PhoneNumberList(
            phone_numbers=phone_number_responses,
            total=len(phone_number_responses)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", response_model=CreatePhoneNumberResponse)
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
        
        return CreatePhoneNumberResponse(
            id=phone_number.id,
            phone_number=phone_number.phone_number,
            label=phone_number.label,
            status=phone_number.status,
            created_at=phone_number.created_at,
            message="Phone number created successfully"
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{phone_number_id}", response_model=PhoneNumberResponse)
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
        
        return PhoneNumberResponse(
            id=phone_number.id,
            tenant_id=phone_number.tenant_id,
            phone_number=phone_number.phone_number,
            label=phone_number.label,
            status=phone_number.status,
            assistant_id=phone_number.assistant_id,
            twilio_phone_number_sid=phone_number.twilio_phone_number_sid,
            created_at=phone_number.created_at,
            updated_at=phone_number.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{phone_number_id}", response_model=PhoneNumberResponse)
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
        
        return PhoneNumberResponse(
            id=phone_number.id,
            tenant_id=phone_number.tenant_id,
            phone_number=phone_number.phone_number,
            label=phone_number.label,
            status=phone_number.status,
            assistant_id=phone_number.assistant_id,
            twilio_phone_number_sid=phone_number.twilio_phone_number_sid,
            created_at=phone_number.created_at,
            updated_at=phone_number.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{phone_number_id}")
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