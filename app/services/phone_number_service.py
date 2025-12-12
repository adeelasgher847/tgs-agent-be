"""
Simple Phone Number Service
"""

from sqlalchemy.orm import Session
from app.models.phone_number import PhoneNumber
from app.schemas.phone_number import PhoneNumberCreate, PhoneNumberUpdate
from typing import List, Optional
import uuid
from datetime import datetime

class PhoneNumberService:
    """Simple phone number service"""
    
    def create_phone_number(self, db: Session, phone_number_data: PhoneNumberCreate) -> PhoneNumber:
        """Create a new phone number"""
        # Check if phone number already exists within the same tenant
        existing = db.query(PhoneNumber).filter(
            PhoneNumber.phone_number == phone_number_data.phone_number,
            PhoneNumber.tenant_id == phone_number_data.tenant_id
        ).first()
        
        if existing:
            raise ValueError(f"Phone number {phone_number_data.phone_number} already exists in this tenant")
        
        # Create phone number
        phone_number = PhoneNumber(**phone_number_data.dict())
        db.add(phone_number)
        db.commit()
        db.refresh(phone_number)
        
        return phone_number
    
    def get_phone_numbers(self, db: Session, tenant_id: uuid.UUID) -> List[PhoneNumber]:
        """Get all phone numbers for a tenant"""
        return db.query(PhoneNumber).filter(PhoneNumber.tenant_id == tenant_id).all()
    
    def get_phone_number_by_id(self, db: Session, phone_number_id: uuid.UUID, tenant_id: uuid.UUID) -> Optional[PhoneNumber]:
        """Get phone number by ID"""
        return db.query(PhoneNumber).filter(
            PhoneNumber.id == phone_number_id,
            PhoneNumber.tenant_id == tenant_id
        ).first()
    
    def update_phone_number(self, db: Session, phone_number_id: uuid.UUID, tenant_id: uuid.UUID, update_data: PhoneNumberUpdate) -> Optional[PhoneNumber]:
        """Update phone number"""
        phone_number = self.get_phone_number_by_id(db, phone_number_id, tenant_id)
        
        if not phone_number:
            return None
        
        # Update fields
        for field, value in update_data.dict(exclude_unset=True).items():
            setattr(phone_number, field, value)
        
        phone_number.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(phone_number)
        
        return phone_number
    
    def delete_phone_number(self, db: Session, phone_number_id: uuid.UUID, tenant_id: uuid.UUID) -> bool:
        """Delete phone number"""
        phone_number = self.get_phone_number_by_id(db, phone_number_id, tenant_id)
        
        if not phone_number:
            return False
        
        db.delete(phone_number)
        db.commit()
        
        return True
    
    def import_twilio_phone_number(
        self, 
        db: Session, 
        phone_number: str,
        label: Optional[str],
        tenant_id: uuid.UUID,
        twilio_account_sid: str,
        twilio_auth_token: str
    ) -> PhoneNumber:
        """
        Import a Twilio phone number with custom credentials
        
        Args:
            db: Database session
            phone_number: Phone number in E.164 format
            label: Optional label
            tenant_id: Tenant ID
            twilio_account_sid: Twilio Account SID
            twilio_auth_token: Twilio Auth Token (will be encrypted)
            
        Returns:
            Created PhoneNumber object
        """
        # Check if phone number already exists within the same tenant
        existing = db.query(PhoneNumber).filter(
            PhoneNumber.phone_number == phone_number,
            PhoneNumber.tenant_id == tenant_id
        ).first()
        
        if existing:
            raise ValueError(f"Phone number {phone_number} already exists in this tenant")
        
        # Encrypt auth token before storing
        from app.core.security import encrypt_api_key
        encrypted_auth_token = encrypt_api_key(twilio_auth_token)
        
        # Create phone number with Twilio credentials
        phone_number_obj = PhoneNumber(
            phone_number=phone_number,
            label=label,
            tenant_id=tenant_id,
            status="active",
            twilio_account_sid=twilio_account_sid,
            twilio_auth_token=encrypted_auth_token
        )
        
        db.add(phone_number_obj)
        db.commit()
        db.refresh(phone_number_obj)
        
        return phone_number_obj

# Create service instance
phone_number_service = PhoneNumberService()
