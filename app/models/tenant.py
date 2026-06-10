from sqlalchemy import Column, String, DateTime, Numeric, Index, text, ForeignKey, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base_class import Base

class Tenant(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String, index=True, nullable=False)
    schema_name = Column(String, unique=False, nullable=False)
    status = Column(String, nullable=False, default="pending_payment")  # pending_payment, active, inactive
    stripe_customer_id = Column(String, nullable=True, index=True)
    stripe_subscription_id = Column(String, nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True, index=True)
    credits = Column(Numeric(10, 4), default=0, nullable=False)  # Float credits with 4 decimal precision
    parent_workspace_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("tenant.id", ondelete="SET NULL"), 
        nullable=True
    )
    workspace_type = Column(
        String, 
        nullable=False, 
        server_default="standalone"
    )
    
    # Relationships
    users = relationship("User", secondary="user_tenant_association", back_populates="tenants") 
    agents = relationship("Agent", back_populates="tenant")
    call_sessions = relationship("CallSession", back_populates="tenant")
    call_logs = relationship("CallLog", back_populates="tenant")
    phone_numbers = relationship("PhoneNumber", back_populates="tenant")
    business_hours = relationship("BusinessHours", back_populates="tenant")
    blocked_slots = relationship("BlockedSlot", back_populates="tenant")
    appointments = relationship("Appointment", back_populates="tenant")
    transfer_routes = relationship("TransferRoute", back_populates="tenant")
    api_keys = relationship("Apikey", back_populates="tenant", cascade="all, delete-orphan")

    parent_workspace = relationship("Tenant", remote_side=[id], backref="sub_accounts")
    branding_config = relationship("BrandingConfig", uselist=False, back_populates="tenant", cascade="all, delete-orphan")
    pricing_config = relationship("PricingConfig", uselist=False, back_populates="tenant", cascade="all, delete-orphan")
    rbac_roles = relationship("RbacRole", back_populates="tenant", cascade="all, delete-orphan")
    usage_records = relationship("UsageRecords", back_populates="tenant", cascade="all, delete-orphan")

    __table_args__ = (
        Index(
            "uq_tenant_name_active",
            "name",
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
            sqlite_where=text("deleted_at IS NULL"),
        ),
        Index("idx_tenants_parent_workspace_id", "parent_workspace_id"),
        CheckConstraint(
            "workspace_type IN ('agency', 'sub_account', 'standalone')", 
            name="chk_tenant_workspace_type"
        ),
    )
