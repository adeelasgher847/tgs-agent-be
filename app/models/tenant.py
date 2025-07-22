from sqlalchemy import Column, Integer, String
from app.db.base_class import Base

class Tenant(Base):
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    schema_name = Column(String, unique=True, nullable=False) 