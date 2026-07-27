"""
CRM Configuration Service
"""

from sqlalchemy.orm import Session
from fastapi import HTTPException
from typing import Optional, List
import uuid
import json

from app.models.tenant_crm_config import CRMConfig
from app.core.security import encrypt_api_key, decrypt_api_key
from app.schemas.crm_config import CRMConfigCreate, CRMConfigUpdate


class CRMConfigService:
    """Service for managing global CRM configurations"""

    @staticmethod
    def create_crm_config(
        db: Session,
        crm_config_data: CRMConfigCreate,
        created_by: uuid.UUID
    ) -> CRMConfig:
        """Create a new global CRM configuration"""
        # Check if CRM config already exists for this CRM type
        existing = db.query(CRMConfig).filter(
            CRMConfig.crm_type == crm_config_data.crm_type.lower()
        ).first()
        
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"CRM configuration for {crm_config_data.crm_type} already exists"
            )
        
        # Encrypt API key (if provided, optional for ClickUp OAuth)
        if crm_config_data.api_key:
            encrypted_api_key = encrypt_api_key(crm_config_data.api_key)
        else:
            # For ClickUp OAuth, api_key can be empty initially
            # It will be set after OAuth callback
            if crm_config_data.crm_type.lower() == "clickup":
                encrypted_api_key = ""  # Will be set after OAuth
            else:
                raise HTTPException(
                    status_code=400,
                    detail="api_key is required for this CRM type"
                )
        
        # Encrypt sensitive fields in additional_config (like api_token for Trello)
        additional_config_encrypted = None
        if crm_config_data.additional_config:
            additional_config_encrypted = crm_config_data.additional_config.copy()
            
            # Encrypt api_token if present (for Trello)
            if "api_token" in additional_config_encrypted and additional_config_encrypted["api_token"]:
                additional_config_encrypted["api_token"] = encrypt_api_key(additional_config_encrypted["api_token"])
            
            # Encrypt client_secret if present (for ClickUp OAuth)
            if "client_secret" in additional_config_encrypted and additional_config_encrypted["client_secret"]:
                additional_config_encrypted["client_secret"] = encrypt_api_key(additional_config_encrypted["client_secret"])
            
            # Serialize to JSON string
            additional_config_json = json.dumps(additional_config_encrypted)
        else:
            additional_config_json = None
        
        crm_config = CRMConfig(
            crm_type=crm_config_data.crm_type.lower(),
            encrypted_api_key=encrypted_api_key,
            container_id=crm_config_data.container_id,
            container_url=crm_config_data.container_url,
            additional_config=additional_config_json,
            created_by=created_by
        )
        
        db.add(crm_config)
        db.commit()
        db.refresh(crm_config)
        
        return crm_config

    @staticmethod
    def get_crm_config_by_id(db: Session, crm_config_id: uuid.UUID) -> Optional[CRMConfig]:
        """Get CRM config by ID"""
        return db.query(CRMConfig).filter(CRMConfig.id == crm_config_id).first()

    @staticmethod
    def get_crm_config_by_type(db: Session, crm_type: str) -> Optional[CRMConfig]:
        """Get CRM config by CRM type"""
        return db.query(CRMConfig).filter(
            CRMConfig.crm_type == crm_type.lower()
        ).first()

    @staticmethod
    def get_all_crm_configs(db: Session) -> List[CRMConfig]:
        """Get all global CRM configs (all 4 CRMs)"""
        return db.query(CRMConfig).all()

    @staticmethod
    def update_crm_config(
        db: Session,
        crm_config_id: uuid.UUID,
        update_data: CRMConfigUpdate
    ) -> CRMConfig:
        """Update CRM configuration"""
        crm_config = db.query(CRMConfig).filter(CRMConfig.id == crm_config_id).first()
        if not crm_config:
            raise HTTPException(status_code=404, detail="CRM configuration not found")
        
        if update_data.api_key:
            crm_config.encrypted_api_key = encrypt_api_key(update_data.api_key)
        
        if update_data.container_id is not None:
            crm_config.container_id = update_data.container_id
        
        if update_data.container_url is not None:
            crm_config.container_url = update_data.container_url
        
        if update_data.additional_config is not None:
            # Encrypt sensitive fields in additional_config (like api_token for Trello)
            additional_config_encrypted = update_data.additional_config.copy()
            
            # Encrypt api_token if present (for Trello)
            if "api_token" in additional_config_encrypted and additional_config_encrypted["api_token"]:
                additional_config_encrypted["api_token"] = encrypt_api_key(additional_config_encrypted["api_token"])
            
            # Encrypt client_secret if present (for ClickUp OAuth)
            if "client_secret" in additional_config_encrypted and additional_config_encrypted["client_secret"]:
                additional_config_encrypted["client_secret"] = encrypt_api_key(additional_config_encrypted["client_secret"])
            
            crm_config.additional_config = json.dumps(additional_config_encrypted)
        
        db.commit()
        db.refresh(crm_config)
        
        return crm_config

    @staticmethod
    def delete_crm_config(db: Session, crm_config_id: uuid.UUID) -> bool:
        """Delete CRM configuration"""
        crm_config = db.query(CRMConfig).filter(CRMConfig.id == crm_config_id).first()
        if not crm_config:
            raise HTTPException(status_code=404, detail="CRM configuration not found")
        
        db.delete(crm_config)
        db.commit()
        
        return True

