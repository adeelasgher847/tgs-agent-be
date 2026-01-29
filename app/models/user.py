from sqlalchemy import Column, String, Table, ForeignKey, DateTime, Boolean, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base_class import Base

user_tenant_association = Table(
    'user_tenant_association', Base.metadata,
    Column('user_id', UUID(as_uuid=True), ForeignKey('user.id')),
    Column('tenant_id', UUID(as_uuid=True), ForeignKey('tenant.id')),
    Column('is_creator', Boolean, nullable=False, default=False),
    Column('role_id', UUID(as_uuid=True), ForeignKey('role.id'), nullable=True)
)

class User(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    phone = Column(String, nullable=True)  # Optional field
    hashed_password = Column(String, nullable=False)
    join_date = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    current_tenant_id = Column(UUID(as_uuid=True), ForeignKey('tenant.id'), nullable=True)
        # OAuth provider fields (for Google, etc.)
    provider = Column(String, nullable=True, index=True)
    provider_user_id = Column(String, nullable=True, index=True)
    provider_profile = Column(JSON, nullable=True)  # JSON as string
    
    tenants = relationship("Tenant", secondary=user_tenant_association, back_populates="users")
    current_tenant = relationship("Tenant", foreign_keys=[current_tenant_id])
    
    # Back references for audit trail
    created_agents = relationship("Agent", foreign_keys="Agent.created_by", back_populates="creator")
    updated_agents = relationship("Agent", foreign_keys="Agent.updated_by", back_populates="updater")
    
    # Password reset tokens
    password_reset_tokens = relationship("PasswordResetToken", back_populates="user", cascade="all, delete-orphan")
    
    # Call sessions
    call_sessions = relationship("CallSession", back_populates="user")
    transcript_messages = relationship("TranscriptMessage", back_populates="user")
    
    # Scheduled calls board (one board per user, shared by all their tenants)
    scheduled_call = relationship("ScheduledCall", back_populates="user", uselist=False)
    
    # Subscriptions
    subscriptions = relationship("Subscription", back_populates="user", uselist=True)