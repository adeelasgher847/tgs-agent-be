"""
Model API endpoints
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.api.deps import get_db, require_tenant
from app.schemas.model import ModelCreate, ModelUpdate, ModelResponse, ModelList
from app.services.model_service import model_service
from app.utils.response import create_success_response
from app.schemas.base import SuccessResponse
import uuid

router = APIRouter()


@router.post("/", response_model=SuccessResponse[ModelResponse])
async def create_model(
    model_data: ModelCreate,
    user = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Create a new AI model
    
    - **provider_id**: ID of the provider this model belongs to
    - **model_name**: Name of the model (e.g., gpt-4, gemini-pro)
    - **api_key**: Model-specific API key for authentication
    - **description**: Model description including free tokens, efficiency, pricing details
    - **system_prompt**: Default system prompt for the model
    - **temperature**: Temperature setting (0-100)
    - **max_tokens**: Maximum tokens for responses
    - **archive**: Whether the model is archived (default: True)
    """
    try:
        model = model_service.create_model(db, model_data)
        return create_success_response(
            model,
            "Model created successfully"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create model: {str(e)}")


@router.get("/", response_model=SuccessResponse[ModelList])
async def get_models(
    skip: int = Query(0, ge=0, description="Number of models to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of models to return"),
    provider_id: uuid.UUID = Query(None, description="Filter by provider ID"),
    active_only: bool = Query(False, description="Return only active (non-archived) models"),
    user = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Get all models with pagination and filtering
    
    - **skip**: Number of models to skip (for pagination)
    - **limit**: Maximum number of models to return
    - **provider_id**: Filter models by provider ID
    - **active_only**: If true, return only active (non-archived) models
    """
    try:
        if provider_id:
            models = model_service.get_models_by_provider_safe(db, provider_id)
        elif active_only:
            models = model_service.get_active_models_safe(db)
        else:
            models = model_service.get_models_safe(db, skip, limit)
        
        return create_success_response(
            {
                "models": models,
                "total": len(models),
                "page": skip // limit + 1 if limit > 0 else 1,
                "size": limit
            },
            "Models retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")


@router.get("/{model_id}", response_model=SuccessResponse[ModelResponse])
async def get_model(
    model_id: uuid.UUID,
    user = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Get a specific model by ID
    """
    model = model_service.get_model_by_id_safe(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return create_success_response(
        model,
        "Model retrieved successfully"
    )


@router.put("/{model_id}", response_model=SuccessResponse[ModelResponse])
async def update_model(
    model_id: uuid.UUID,
    model_data: ModelUpdate,
    user = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Update a model
    
    - **model_name**: New model name (optional)
    - **api_key**: New API key (optional)
    - **description**: New description (optional)
    - **system_prompt**: New system prompt (optional)
    - **temperature**: New temperature setting (optional)
    - **max_tokens**: New max tokens setting (optional)
    - **archive**: New archive status (optional)
    """
    try:
        model = model_service.update_model(db, model_id, model_data)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return create_success_response(
            model,
            "Model updated successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update model: {str(e)}")


@router.delete("/{model_id}", response_model=SuccessResponse[dict])
async def delete_model(
    model_id: uuid.UUID,
    hard_delete: bool = Query(False, description="Perform hard delete instead of soft delete"),
    user = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Delete a model
    
    - **hard_delete**: If true, permanently delete the model. If false, soft delete (set archive=True)
    """
    try:
        if hard_delete:
            success = model_service.hard_delete_model(db, model_id)
        else:
            success = model_service.delete_model(db, model_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Model not found")
        
        action = "permanently deleted" if hard_delete else "archived"
        return create_success_response(
            {"model_id": str(model_id)},
            f"Model {action} successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


@router.get("/provider/{provider_id}", response_model=SuccessResponse[ModelList])
async def get_models_by_provider(
    provider_id: uuid.UUID,
    user = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Get all models for a specific provider
    """
    try:
        models = model_service.get_models_by_provider_safe(db, provider_id)
        return create_success_response(
            {
                "models": models,
                "total": len(models),
                "page": 1,
                "size": len(models)
            },
            f"Models for provider {provider_id} retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get models by provider: {str(e)}")
