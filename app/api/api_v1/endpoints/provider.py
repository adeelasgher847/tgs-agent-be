"""
Provider API endpoints
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.api.deps import get_db, require_tenant
from app.schemas.provider import ProviderCreate, ProviderUpdate, ProviderResponse, ProviderList
from app.services.provider_service import provider_service
from app.utils.response import create_success_response
from app.schemas.base import SuccessResponse
import uuid

router = APIRouter()


@router.post("/", response_model=SuccessResponse[ProviderResponse])
async def create_provider(
    provider_data: ProviderCreate,
    user = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Create a new AI provider
    
    - **name**: Provider name (e.g., OpenAI, Google, Anthropic)
    - **api_key**: API key for the provider (optional)
    - **is_active**: Whether the provider is active (default: True)
    """
    try:
        provider = provider_service.create_provider(db, provider_data)
        return create_success_response(
            provider,
            "Provider created successfully"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create provider: {str(e)}")


@router.get("/", response_model=SuccessResponse[ProviderList])
async def get_providers(
    skip: int = Query(0, ge=0, description="Number of providers to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of providers to return"),
    active_only: bool = Query(False, description="Return only active providers"),
    user = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Get all providers with pagination
    
    - **skip**: Number of providers to skip (for pagination)
    - **limit**: Maximum number of providers to return
    - **active_only**: If true, return only active providers
    """
    try:
        if active_only:
            providers = provider_service.get_active_providers(db)
        else:
            providers = provider_service.get_all_providers(db, skip, limit)
        
        return create_success_response(
            {
                "providers": providers,
                "total": len(providers),
                "page": skip // limit + 1 if limit > 0 else 1,
                "size": limit
            },
            "Providers retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get providers: {str(e)}")


@router.get("/{provider_id}", response_model=SuccessResponse[ProviderResponse])
async def get_provider(
    provider_id: uuid.UUID,
    user = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Get a specific provider by ID
    """
    provider = provider_service.get_provider_by_id(db, provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    return create_success_response(
        provider,
        "Provider retrieved successfully"
    )


@router.put("/{provider_id}", response_model=SuccessResponse[ProviderResponse])
async def update_provider(
    provider_id: uuid.UUID,
    provider_data: ProviderUpdate,
    user = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Update a provider
    
    - **name**: New provider name (optional)
    - **api_key**: New API key (optional)
    - **is_active**: New active status (optional)
    """
    try:
        provider = provider_service.update_provider(db, provider_id, provider_data)
        if not provider:
            raise HTTPException(status_code=404, detail="Provider not found")
        
        return create_success_response(
            provider,
            "Provider updated successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update provider: {str(e)}")


@router.delete("/{provider_id}", response_model=SuccessResponse[dict])
async def delete_provider(
    provider_id: uuid.UUID,
    hard_delete: bool = Query(False, description="Perform hard delete instead of soft delete"),
    user = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Delete a provider
    
    - **hard_delete**: If true, permanently delete the provider. If false, soft delete (set is_active=False)
    """
    try:
        if hard_delete:
            success = provider_service.hard_delete_provider(db, provider_id)
        else:
            success = provider_service.delete_provider(db, provider_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Provider not found")
        
        action = "permanently deleted" if hard_delete else "deactivated"
        return create_success_response(
            {"provider_id": str(provider_id)},
            f"Provider {action} successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete provider: {str(e)}")
