from sqlalchemy import Column, Integer, String, Table, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base_class import Base
from datetime import datetime

user_tenant_association = Table(
    'user_tenant_association', Base.metadata,
    Column('user_id', Integer, ForeignKey('user.id')),
    Column('tenant_id', Integer, ForeignKey('tenant.id'))
)

class User(Base):
    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    phone = Column(String, nullable=True)  # Optional field
    hashed_password = Column(String, nullable=False)
    tenant_id = Column(Integer, ForeignKey('tenant.id'), nullable=True)
    join_date = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    tenants = relationship("Tenant", secondary=user_tenant_association, back_populates="users") 