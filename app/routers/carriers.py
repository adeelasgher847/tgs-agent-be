"""
Carrier Router
API endpoints for carrier management
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import uuid

from app.api.deps import get_db, require_tenant
from app.models.user import User
from app.schemas.carrier import CarrierCreate, CarrierUpdate, CarrierResponse, CarrierList
from app.schemas.base import SuccessResponse
from app.services.carrier_service import carrier_service
from app.utils.response import create_success_response
from app.core.logger import logger

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
