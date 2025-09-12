from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base_class import Base

class Invite(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    email = Column(String, nullable=False, index=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey('tenant.id'), nullable=False)
    invited_by = Column(UUID(as_uuid=True), ForeignKey('user.id'), nullable=False)
    token = Column(String, unique=True, nullable=False, index=True)
    status = Column(String, default="PENDING", nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    accepted_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    tenant = relationship("Tenant")
    inviter = relationship("User", foreign_keys=[invited_by])
