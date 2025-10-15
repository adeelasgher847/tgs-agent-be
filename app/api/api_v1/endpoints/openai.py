"""
OpenAI API endpoints for text generation and chat completion testing
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.api.deps import get_db, require_tenant
from app.schemas.openai import OpenAITestRequest, OpenAITestResponse, OpenAIChatRequest, OpenAIChatResponse
from app.schemas.base import SuccessResponse
from app.services.openai_service import openai_service
from app.services.model_service import model_service
from app.core.security import decrypt_api_key
from app.core.config import settings
from openai import OpenAI
import uuid

router = APIRouter()


@router.post("/test-text-generation", response_model=SuccessResponse[OpenAITestResponse])
async def test_openai_text_generation(
    request: OpenAITestRequest,
    user = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Test OpenAI text-to-text generation using a model from the database
    
    - **model_id**: ID of the OpenAI model to use for generation
    - **prompt**: Input prompt for text generation
    - **system_prompt**: Optional system prompt to set context
    - **temperature**: Temperature setting (0.0 to 2.0, default: 0.7)
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
                print(f"Failed to decrypt model API key: {e}")
                # If decryption fails, use global key
                pass
        
        # Generate text using OpenAI with model-specific API key
        response = openai_service.generate_text(
            prompt=request.prompt,
            system_prompt=system_prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key
        )
        
        return SuccessResponse(
            data=OpenAITestResponse(
                content=response["content"],
                model_name=response["model"],
                response_time=response["response_time"],
                usage=response["usage"],
                finish_reason=response["finish_reason"]
            ),
            message="OpenAI text generation completed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate text: {str(e)}")


@router.post("/test-chat-completion", response_model=SuccessResponse[OpenAIChatResponse])
async def test_openai_chat_completion(
    request: OpenAIChatRequest,
    user = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Test OpenAI chat completion using a model from the database
    
    - **model_id**: ID of the OpenAI model to use for generation
    - **messages**: List of conversation messages with 'role' and 'content'
    - **system_prompt**: Optional system prompt to set context
    - **temperature**: Temperature setting (0.0 to 2.0, default: 0.7)
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
                print(f"Failed to decrypt model API key: {e}")
                # If decryption fails, use global key
                pass
        
        # Validate messages format
        for message in request.messages:
            if "role" not in message or "content" not in message:
                raise HTTPException(status_code=400, detail="Each message must have 'role' and 'content' fields")
            if message["role"] not in ["user", "assistant", "system"]:
                raise HTTPException(status_code=400, detail="Message role must be 'user', 'assistant', or 'system'")
        
        # Generate chat completion using OpenAI with model-specific API key
        response = openai_service.chat_completion(
            messages=request.messages,
            system_prompt=system_prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key
        )
        
        return SuccessResponse(
            data=OpenAIChatResponse(
                content=response["content"],
                model_name=response["model"],
                response_time=response["response_time"],
                usage=response["usage"],
                finish_reason=response["finish_reason"]
            ),
            message="OpenAI chat completion completed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to complete chat: {str(e)}")


@router.get("/models/openai", response_model=SuccessResponse[list])
async def get_openai_models(
    user = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Get all OpenAI models from the database
    """
    try:
        # Get all active models
        models = model_service.get_active_models(db)
        
        # Filter for OpenAI models (assuming model names contain 'gpt' or 'openai')
        openai_models = []
        for model in models:
            if 'gpt' in model.model_name.lower() or 'openai' in model.model_name.lower():
                openai_models.append({
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
            data=openai_models,
            message=f"Found {len(openai_models)} OpenAI models"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get OpenAI models: {str(e)}")


@router.get("/available-models", response_model=SuccessResponse[dict])
async def get_available_openai_models(
    user = Depends(require_tenant)
):
    """
    Get list of available OpenAI models from OpenAI API
    
    This endpoint checks which models are available with your OpenAI API key.
    Use this to verify model names before creating them in the database.
    """
    try:
        if not settings.OPENAI_API_KEY:
            return SuccessResponse(
                data={
                    "available": False,
                    "message": "OpenAI API key not configured in settings",
                    "recommended_models": [
                        {
                            "name": "gpt-4o",
                            "description": "Latest GPT-4 optimized - Fast and powerful",
                            "use_case": "Best for voice agents and complex tasks",
                            "pricing": "$$"
                        },
                        {
                            "name": "gpt-4o-mini",
                            "description": "Lightweight GPT-4 - Affordable and fast",
                            "use_case": "Great for conversations and quick responses",
                            "pricing": "$"
                        },
                        {
                            "name": "gpt-4-turbo",
                            "description": "Advanced GPT-4 with extended context",
                            "use_case": "Best for long conversations",
                            "pricing": "$$$"
                        },
                        {
                            "name": "gpt-3.5-turbo",
                            "description": "Fast and efficient - Most cost-effective",
                            "use_case": "Perfect for basic conversations",
                            "pricing": "$"
                        }
                    ]
                },
                message="OpenAI API key not configured. Here are recommended models."
            )
        
        # Create OpenAI client
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Get available models
        models = client.models.list()
        
        # Filter only chat completion models (GPT models)
        chat_models = []
        for model in models.data:
            if 'gpt' in model.id.lower():
                chat_models.append({
                    "id": model.id,
                    "owned_by": model.owned_by,
                    "created": model.created
                })
        
        # Sort by name
        chat_models = sorted(chat_models, key=lambda x: x['id'])
        
        # Recommended models for voice agents
        recommended_for_voice = [
            {
                "name": "gpt-4o-mini",
                "reason": "Fast, affordable, great for voice",
                "temperature": "0.7-0.8",
                "max_tokens": "500-1000"
            },
            {
                "name": "gpt-4o",
                "reason": "Best quality, fast responses",
                "temperature": "0.7-0.9",
                "max_tokens": "1000-1500"
            },
            {
                "name": "gpt-3.5-turbo",
                "reason": "Budget-friendly, quick",
                "temperature": "0.7-0.8",
                "max_tokens": "500-800"
            }
        ]
        
        return SuccessResponse(
            data={
                "available": True,
                "total_models": len(chat_models),
                "chat_models": chat_models,
                "recommended_for_voice": recommended_for_voice,
                "model_naming_guide": {
                    "correct_examples": [
                        "gpt-4o",
                        "gpt-4o-mini",
                        "gpt-4-turbo",
                        "gpt-3.5-turbo"
                    ],
                    "incorrect_examples": [
                        "gpt-5 (doesn't exist yet)",
                        "gpt4 (missing hyphen)",
                        "chatgpt (not an API model)",
                        "gpt-4o-mini-2 (wrong version)"
                    ]
                }
            },
            message=f"Found {len(chat_models)} available OpenAI models"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to fetch available models: {str(e)}"
        )


@router.post("/validate-model-name", response_model=SuccessResponse[dict])
async def validate_model_name(
    model_name: str,
    user = Depends(require_tenant)
):
    """
    Validate if a model name exists in OpenAI
    
    Test if a specific model name is valid before creating it in database.
    This makes a minimal API call to check if the model works.
    
    **Query Parameter:**
    - **model_name**: Name of the model to validate (e.g., gpt-4o, gpt-3.5-turbo)
    
    **Example:** `/api/v1/openai/validate-model-name?model_name=gpt-4o`
    """
    try:
        if not settings.OPENAI_API_KEY:
            raise HTTPException(
                status_code=400, 
                detail="OpenAI API key not configured in settings"
            )
        
        # Create OpenAI client
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Try to make a minimal API call with this model
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            
            return SuccessResponse(
                data={
                    "valid": True,
                    "model_name": model_name,
                    "test_successful": True,
                    "model_info": {
                        "model": response.model,
                        "can_be_used": True
                    },
                    "message": f"✅ Model '{model_name}' is valid and working!",
                    "recommendation": "You can safely use this model name when creating models in database."
                },
                message=f"Model '{model_name}' validated successfully"
            )
            
        except Exception as e:
            error_message = str(e)
            
            if "does not exist" in error_message or "not found" in error_message:
                return SuccessResponse(
                    data={
                        "valid": False,
                        "model_name": model_name,
                        "test_successful": False,
                        "error": "Model does not exist",
                        "message": f"❌ Model '{model_name}' does not exist in OpenAI",
                        "suggestion": "Check correct model names at: https://platform.openai.com/docs/models",
                        "recommended_alternatives": [
                            "gpt-4o",
                            "gpt-4o-mini",
                            "gpt-4-turbo",
                            "gpt-3.5-turbo"
                        ]
                    },
                    message=f"Model '{model_name}' is invalid"
                )
            elif "quota" in error_message.lower() or "rate" in error_message.lower():
                return SuccessResponse(
                    data={
                        "valid": True,
                        "model_name": model_name,
                        "test_successful": False,
                        "message": f"⚠️ Model '{model_name}' exists but quota/rate limit exceeded",
                        "note": "Model is valid but you've hit API limits"
                    },
                    message=f"Model '{model_name}' exists but has quota issues"
                )
            else:
                return SuccessResponse(
                    data={
                        "valid": False,
                        "model_name": model_name,
                        "test_successful": False,
                        "error": error_message,
                        "message": f"❌ Error testing model '{model_name}': {error_message}"
                    },
                    message=f"Error validating model '{model_name}'"
                )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to validate model name: {str(e)}"
        )

