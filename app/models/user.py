from sqlalchemy import Column, Integer, String, Table, ForeignKey
from sqlalchemy.orm import relationship
from app.db.base_class import Base

user_tenant_association = Table(
    'user_tenant_association', Base.metadata,
    Column('user_id', Integer, ForeignKey('user.id')),
    Column('tenant_id', Integer, ForeignKey('tenant.id'))
)

class User(Base):
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    tenant_id = Column(Integer, ForeignKey('tenant.id'), nullable=True)  
    tenants = relationship("Tenant", secondary=user_tenant_association, back_populates="users") 