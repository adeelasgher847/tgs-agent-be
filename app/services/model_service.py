"""
Model Service
Handles model CRUD operations
"""

from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.models.model import Model
from app.models.provider import Provider
from app.schemas.model import ModelCreate, ModelUpdate
from app.core.security import encrypt_api_key, decrypt_api_key, is_api_key_encrypted
import uuid
from app.services.pricing_service import PricingService

pricing_service = PricingService()

class ModelService:
    """Service for model operations"""
    
    def create_model(self, db: Session, model_data: ModelCreate) -> Model:
        """Create a new model"""
        # Verify provider exists
        provider = db.query(Provider).filter(Provider.id == model_data.provider_id).first()
        if not provider:
            raise ValueError(f"Provider with ID '{model_data.provider_id}' not found")
        
        try:
            # Encrypt API key before storing
            encrypted_api_key = encrypt_api_key(model_data.api_key) if model_data.api_key else None
            
            model = Model(
                provider_id=model_data.provider_id,
                model_name=model_data.model_name,
                api_key=encrypted_api_key,
                description=model_data.description,
                system_prompt=model_data.system_prompt,
                temperature=model_data.temperature,
                max_tokens=model_data.max_tokens,
                archive=model_data.archive
            )
            db.add(model)
            db.commit()
            db.refresh(model)
            return model
        except IntegrityError as e:
            db.rollback()
            raise ValueError(f"Failed to create model: {str(e)}")
    
    def get_model_by_id(self, db: Session, model_id: uuid.UUID) -> Optional[Model]:
        """Get model by ID"""
        return db.query(Model).filter(Model.id == model_id).first()
    
    def get_model_by_name(self, db: Session, model_name: str) -> Optional[Model]:
        """Get model by name"""
        return db.query(Model).filter(Model.model_name == model_name).first()
    
    def get_models_by_provider(self, db: Session, provider_id: uuid.UUID) -> List[Model]:
        """Get all models for a specific provider"""
        return db.query(Model).filter(Model.provider_id == provider_id).all()
    
    def get_all_models(self, db: Session, skip: int = 0, limit: int = 100) -> List[Model]:
        """Get all models with pagination"""
        return db.query(Model).offset(skip).limit(limit).all()
    
    def get_active_models(self, db: Session) -> List[Model]:
        """Get all active (non-archived) models"""
        return db.query(Model).filter(Model.archive == False).all()
    
    def update_model(self, db: Session, model_id: uuid.UUID, model_data: ModelUpdate) -> Optional[Model]:
        """Update a model"""
        model = self.get_model_by_id(db, model_id)
        if not model:
            return None
        
        update_data = model_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            if field == 'api_key' and value:
                # Encrypt API key before storing
                value = encrypt_api_key(value)
            setattr(model, field, value)
        
        db.commit()
        db.refresh(model)
        return model
    
    def delete_model(self, db: Session, model_id: uuid.UUID) -> bool:
        """Delete a model (soft delete by setting archive=True)"""
        model = self.get_model_by_id(db, model_id)
        if not model:
            return False
        
        model.archive = True
        db.commit()
        return True
    
    def hard_delete_model(self, db: Session, model_id: uuid.UUID) -> bool:
        """Hard delete a model"""
        model = self.get_model_by_id(db, model_id)
        if not model:
            return False
        
        db.delete(model)
        db.commit()
        return True
    
    def get_model_with_decrypted_api_key(self, db: Session, model_id: uuid.UUID) -> Optional[dict]:
        """Get model with decrypted API key for use in API calls"""
        model = self.get_model_by_id(db, model_id)
        if not model:
            return None
        
        model_dict = {
            "id": model.id,
            "provider_id": model.provider_id,
            "model_name": model.model_name,
            "description": model.description,
            "system_prompt": model.system_prompt,
            "temperature": model.temperature,
            "max_tokens": model.max_tokens,
            "archive": model.archive,
            "created_at": model.created_at,
            "updated_at": model.updated_at
        }
          
        # Decrypt API key if it exists
        if model.api_key:
            try:
                model_dict["api_key"] = decrypt_api_key(model.api_key)
            except ValueError:
                model_dict["api_key"] = None  # Failed to decrypt
        
        return model_dict

        
    
    def model_to_safe_dict(self, model: Model) -> dict:
        """Convert model to dictionary without API key for safe API responses"""
        data = {
        "id": model.id,
        "provider_id": model.provider_id,
        "model_name": model.model_name,
        "description": model.description,
        "system_prompt": model.system_prompt,
        "temperature": model.temperature,
        "max_tokens": model.max_tokens,
        "archive": model.archive,
        "created_at": model.created_at,
        "updated_at": model.updated_at
    }
        # ✅ Use correct PricingService method
        pricing = pricing_service.get_pricing_for_model(model.model_name)
        if pricing:
         data["pricing"] = {
            "llm_per_min": pricing.get("llm_cost_per_minute"),
            "twilio_per_min": pricing.get("twilio_cost_per_minute"),
            "total_per_min": pricing.get("total_cost_per_minute")
        }
        else:
            data["pricing"] = {"error": "Pricing not found for this model"}
        return data
    
    def get_models_safe(self, db: Session, skip: int = 0, limit: int = 100) -> List[dict]:
        """Get all models as safe dictionaries (no API keys)"""
        models = self.get_all_models(db, skip, limit)
        return [self.model_to_safe_dict(model) for model in models]
    
    def get_active_models_safe(self, db: Session) -> List[dict]:
        """Get active models as safe dictionaries (no API keys)"""
        models = self.get_active_models(db)
        return [self.model_to_safe_dict(model) for model in models]
    
    def get_models_by_provider_safe(self, db: Session, provider_id: uuid.UUID) -> List[dict]:
        """Get models by provider as safe dictionaries (no API keys)"""
        models = self.get_models_by_provider(db, provider_id)
        return [self.model_to_safe_dict(model) for model in models]
    
    def get_model_by_id_safe(self, db: Session, model_id: uuid.UUID) -> Optional[dict]:
        """Get model by ID as safe dictionary (no API key)"""
        model = self.get_model_by_id(db, model_id)
        if not model:
            return None
        return self.model_to_safe_dict(model)


# Create service instance
model_service = ModelService()
