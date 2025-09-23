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
        # Check if phone number already exists
        existing = db.query(PhoneNumber).filter(
            PhoneNumber.phone_number == phone_number_data.phone_number
        ).first()
        
        if existing:
            raise ValueError(f"Phone number {phone_number_data.phone_number} already exists")
        
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

# Create service instance
phone_number_service = PhoneNumberService()
