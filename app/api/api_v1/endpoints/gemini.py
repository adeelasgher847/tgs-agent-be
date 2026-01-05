"""
Gemini API endpoints for text generation testing
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.api.deps import get_db, require_tenant
from app.schemas.gemini import GeminiTestRequest, GeminiTestResponse, GeminiChatRequest, GeminiChatResponse
from app.schemas.base import SuccessResponse
from app.services.gemini_service import gemini_service
from app.services.model_service import model_service
from app.core.security import decrypt_api_key
import uuid
from app.core.logger import logger

router = APIRouter()


@router.post("/test-text-generation", response_model=SuccessResponse[GeminiTestResponse])
async def test_gemini_text_generation(
    request: GeminiTestRequest,
    user = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Test Gemini text-to-text generation using a model from the database
    
    - **model_id**: ID of the Gemini model to use for generation
    - **prompt**: Input prompt for text generation
    - **system_prompt**: Optional system prompt to set context
    - **temperature**: Temperature setting (0.0 to 1.0, default: 0.7)
    - **max_tokens**: Maximum tokens for response (default: 1000)
    """
    try:
        # Get the model from database
        model = model_service.get_model_by_id(db, request.model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Check if model is active
        if model.archive:
            raise HTTPException(status_code=400, detail="Model is archived and cannot be used")
        
        # Get model details
        model_name = model.model_name
        system_prompt = request.system_prompt or model.system_prompt
        temperature = request.temperature or (model.temperature / 100.0 if model.temperature else 0.7)
        max_tokens = request.max_tokens or model.max_tokens or 1000
        
        # Use model-specific API key if available, otherwise use global key
        api_key = None
        if model.api_key:
            try:
                api_key = decrypt_api_key(model.api_key)
            except Exception as e:
                logger.error(f"Failed to decrypt model API key: {e}", exc_info=True)
                # If decryption fails, use global key
                pass
        
        # Generate text using Gemini with model-specific API key
        response = gemini_service.generate_text(
            prompt=request.prompt,
            system_prompt=system_prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key
        )
        
        return SuccessResponse(
            data=GeminiTestResponse(
                content=response["content"],
                model_name=response["model"],
                response_time=response["response_time"],
                usage=response["usage"],
                finish_reason=response["finish_reason"]
            ),
            message="Gemini text generation completed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate text: {str(e)}")


@router.post("/test-chat-completion", response_model=SuccessResponse[GeminiChatResponse])
async def test_gemini_chat_completion(
    request: GeminiChatRequest,
    user = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Test Gemini chat completion using a model from the database
    
    - **model_id**: ID of the Gemini model to use for generation
    - **messages**: List of conversation messages with 'role' and 'content'
    - **system_prompt**: Optional system prompt to set context
    - **temperature**: Temperature setting (0.0 to 1.0, default: 0.7)
    - **max_tokens**: Maximum tokens for response (default: 1000)
    """
    try:
        # Get the model from database
        model = model_service.get_model_by_id(db, request.model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Check if model is active
        if model.archive:
            raise HTTPException(status_code=400, detail="Model is archived and cannot be used")
        
        # Get model details
        model_name = model.model_name
        system_prompt = request.system_prompt or model.system_prompt
        temperature = request.temperature or (model.temperature / 100.0 if model.temperature else 0.7)
        max_tokens = request.max_tokens or model.max_tokens or 1000
        
        # Use model-specific API key if available, otherwise use global key
        api_key = None
        if model.api_key:
            try:
                api_key = decrypt_api_key(model.api_key)
            except Exception as e:
                logger.error(f"Failed to decrypt model API key: {e}", exc_info=True)
                # If decryption fails, use global key
                pass
        
        # Validate messages format
        for message in request.messages:
            if "role" not in message or "content" not in message:
                raise HTTPException(status_code=400, detail="Each message must have 'role' and 'content' fields")
            if message["role"] not in ["user", "assistant", "system"]:
                raise HTTPException(status_code=400, detail="Message role must be 'user', 'assistant', or 'system'")
        
        # Generate chat completion using Gemini with model-specific API key
        response = gemini_service.chat_completion(
            messages=request.messages,
            system_prompt=system_prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key
        )
        
        return SuccessResponse(
            data=GeminiChatResponse(
                content=response["content"],
                model_name=response["model"],
                response_time=response["response_time"],
                usage=response["usage"],
                finish_reason=response["finish_reason"]
            ),
            message="Gemini chat completion completed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to complete chat: {str(e)}")


@router.get("/models/gemini", response_model=SuccessResponse[list])
async def get_gemini_models(
    user = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Get all Gemini models from the database
    """
    try:
        # Get all active models
        models = model_service.get_active_models(db)
        
        # Filter for Gemini models (assuming model names contain 'gemini')
        gemini_models = []
        for model in models:
            if 'gemini' in model.model_name.lower():
                gemini_models.append({
                    "id": str(model.id),
                    "model_name": model.model_name,
                    "description": model.description,
                    "system_prompt": model.system_prompt,
                    "temperature": model.temperature,
                    "max_tokens": model.max_tokens,
                    "provider_id": str(model.provider_id),
                    "created_at": model.created_at,
                    "updated_at": model.updated_at
                })
        
        return SuccessResponse(
            data=gemini_models,
            message=f"Found {len(gemini_models)} Gemini models"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get Gemini models: {str(e)}")
