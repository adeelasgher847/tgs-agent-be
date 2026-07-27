"""
Provider Service
Handles provider CRUD operations
"""

from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.models.provider import Provider
from app.schemas.provider import ProviderCreate, ProviderUpdate
from app.core.security import encrypt_api_key, decrypt_api_key, is_api_key_encrypted
import uuid


class ProviderService:
    """Service for provider operations"""
    
    def create_provider(self, db: Session, provider_data: ProviderCreate) -> Provider:
        """Create a new provider"""
        try:
            # Encrypt API key before storing
            encrypted_api_key = encrypt_api_key(provider_data.api_key) if provider_data.api_key else None
            
            provider = Provider(
                name=provider_data.name,
                api_key=encrypted_api_key,
                is_active=provider_data.is_active
            )
            db.add(provider)
            db.commit()
            db.refresh(provider)
            return provider
        except IntegrityError:
            db.rollback()
            raise ValueError(f"Provider with name '{provider_data.name}' already exists")
    
    def get_provider_by_id(self, db: Session, provider_id: uuid.UUID) -> Optional[Provider]:
        """Get provider by ID"""
        return db.query(Provider).filter(Provider.id == provider_id).first()
    
    def get_provider_by_name(self, db: Session, name: str) -> Optional[Provider]:
        """Get provider by name"""
        return db.query(Provider).filter(Provider.name == name).first()
    
    def get_all_providers(self, db: Session, skip: int = 0, limit: int = 100) -> List[Provider]:
        """Get all providers with pagination"""
        return db.query(Provider).offset(skip).limit(limit).all()
    
    def get_active_providers(self, db: Session) -> List[Provider]:
        """Get all active providers"""
        return db.query(Provider).filter(Provider.is_active == True).all()
    
    def update_provider(self, db: Session, provider_id: uuid.UUID, provider_data: ProviderUpdate) -> Optional[Provider]:
        """Update a provider"""
        provider = self.get_provider_by_id(db, provider_id)
        if not provider:
            return None
        
        update_data = provider_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            if field == 'api_key' and value:
                # Encrypt API key before storing
                value = encrypt_api_key(value)
            setattr(provider, field, value)
        
        db.commit()
        db.refresh(provider)
        return provider
    
    def delete_provider(self, db: Session, provider_id: uuid.UUID) -> bool:
        """Delete a provider (soft delete by setting is_active=False)"""
        provider = self.get_provider_by_id(db, provider_id)
        if not provider:
            return False
        
        provider.is_active = False
        db.commit()
        return True
    
    def hard_delete_provider(self, db: Session, provider_id: uuid.UUID) -> bool:
        """Hard delete a provider"""
        provider = self.get_provider_by_id(db, provider_id)
        if not provider:
            return False
        
        db.delete(provider)
        db.commit()
        return True
    
    def get_provider_with_decrypted_api_key(self, db: Session, provider_id: uuid.UUID) -> Optional[dict]:
        """Get provider with decrypted API key for use in API calls"""
        provider = self.get_provider_by_id(db, provider_id)
        if not provider:
            return None
        
        provider_dict = {
            "id": provider.id,
            "name": provider.name,
            "is_active": provider.is_active,
            "created_at": provider.created_at,
            "updated_at": provider.updated_at
        }
        
        # Decrypt API key if it exists
        if provider.api_key:
            try:
                provider_dict["api_key"] = decrypt_api_key(provider.api_key)
            except ValueError:
                provider_dict["api_key"] = None  # Failed to decrypt
        
        return provider_dict


# Create service instance
provider_service = ProviderService()
