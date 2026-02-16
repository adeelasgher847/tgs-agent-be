"""
Carrier Service Module
Handles carrier management operations
"""

from sqlalchemy.orm import Session
from app.models.carrier import Carrier
from app.schemas.carrier import CarrierCreate, CarrierUpdate
from typing import List, Optional
import uuid
from datetime import datetime
from app.core.security import encrypt_api_key
from app.core.logger import logger


class CarrierService:
    """Service class for handling carrier operations"""
    
    def create_carrier(self, db: Session, carrier_data: CarrierCreate) -> Carrier:
        """Create a new carrier with encrypted SIP credentials (global or tenant-specific)"""
        # Check if carrier name already exists (global or same tenant)
        query = db.query(Carrier).filter(Carrier.name == carrier_data.name)
        
        if carrier_data.tenant_id:
            # Tenant-specific carrier - check within tenant
            query = query.filter(Carrier.tenant_id == carrier_data.tenant_id)
        else:
            # Global carrier - check if global carrier with same name exists
            query = query.filter(Carrier.tenant_id.is_(None))
        
        existing = query.first()
        
        if existing:
            if carrier_data.tenant_id:
                raise ValueError(f"Carrier '{carrier_data.name}' already exists in this tenant")
            else:
                raise ValueError(f"Global carrier '{carrier_data.name}' already exists")
        
        # Encrypt SIP credentials if provided
        encrypted_sip_username = None
        encrypted_sip_password = None
        
        if carrier_data.sip_username:
            encrypted_sip_username = encrypt_api_key(carrier_data.sip_username)
        
        if carrier_data.sip_password:
            encrypted_sip_password = encrypt_api_key(carrier_data.sip_password)
        
        # Create carrier
        carrier = Carrier(
            tenant_id=carrier_data.tenant_id,
            name=carrier_data.name,
            provider=carrier_data.provider,
            status=carrier_data.status,
            description=carrier_data.description,
            sip_username=encrypted_sip_username,
            sip_password=encrypted_sip_password,
            sip_server=carrier_data.sip_server,
            sip_port=carrier_data.sip_port,
            vicidial_carrier_id=carrier_data.vicidial_carrier_id
        )
        
        db.add(carrier)
        db.commit()
        db.refresh(carrier)
        
        logger.info(f"✅ Carrier created: {carrier.name} (ID: {carrier.id})")
        return carrier
    
    def get_carriers(self, db: Session, tenant_id: Optional[uuid.UUID] = None) -> List[Carrier]:
        """Get all carriers (global + tenant-specific if tenant_id provided)"""
        if tenant_id:
            # Return global carriers (tenant_id=None) + tenant-specific carriers
            return db.query(Carrier).filter(
                (Carrier.tenant_id == tenant_id) | (Carrier.tenant_id.is_(None))
            ).all()
        else:
            # Return only global carriers
            return db.query(Carrier).filter(Carrier.tenant_id.is_(None)).all()
    
    def get_carrier_by_id(self, db: Session, carrier_id: uuid.UUID, tenant_id: Optional[uuid.UUID] = None) -> Optional[Carrier]:
        """Get carrier by ID (global or tenant-specific)"""
        query = db.query(Carrier).filter(Carrier.id == carrier_id)
        
        if tenant_id:
            # Return if global (tenant_id=None) or belongs to tenant
            query = query.filter(
                (Carrier.tenant_id == tenant_id) | (Carrier.tenant_id.is_(None))
            )
        else:
            # Return only global carriers
            query = query.filter(Carrier.tenant_id.is_(None))
        
        return query.first()
    
    def update_carrier(self, db: Session, carrier_id: uuid.UUID, tenant_id: Optional[uuid.UUID] = None, update_data: CarrierUpdate = None) -> Optional[Carrier]:
        """Update carrier (global or tenant-specific)"""
        carrier = self.get_carrier_by_id(db, carrier_id, tenant_id)
        
        if not carrier:
            return None
        
        # Update fields
        for field, value in update_data.dict(exclude_unset=True).items():
            # Encrypt SIP credentials if being updated
            if field == "sip_username" and value:
                value = encrypt_api_key(value)
            elif field == "sip_password" and value:
                value = encrypt_api_key(value)
            
            setattr(carrier, field, value)
        
        carrier.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(carrier)
        
        logger.info(f"✅ Carrier updated: {carrier.name} (ID: {carrier.id})")
        return carrier
    
    def delete_carrier(self, db: Session, carrier_id: uuid.UUID, tenant_id: Optional[uuid.UUID] = None) -> bool:
        """Delete carrier (only if not in use by any phone numbers)"""
        carrier = self.get_carrier_by_id(db, carrier_id, tenant_id)
        
        if not carrier:
            return False
        
        # Check if carrier is in use
        from app.models.phone_number import PhoneNumber
        phone_numbers_using_carrier = db.query(PhoneNumber).filter(
            PhoneNumber.carrier_id == carrier_id
        ).count()
        
        if phone_numbers_using_carrier > 0:
            raise ValueError(f"Cannot delete carrier '{carrier.name}': {phone_numbers_using_carrier} phone number(s) are using it")
        
        db.delete(carrier)
        db.commit()
        
        logger.info(f"✅ Carrier deleted: {carrier.name} (ID: {carrier.id})")
        return True


# Global instance
carrier_service = CarrierService()
