from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base_class import Base

class CRMConfig(Base):

    """Global CRM configuration - stores API keys for Monday.com, ClickUp, Jira, Trello.
    All users can select any of these 4 CRMs."""

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # CRM type: "monday" | "clickup" | "jira" | "trello" (unique - only one config per CRM type)
    crm_type = Column(String(20), nullable=False, unique=True, index=True)
    
    # Encrypted API key/token (using JWT encryption)
    encrypted_api_key = Column(String(1000), nullable=False)
    
    # Container info (board_id/list_id/project_id) - optional, can be set later
    container_id = Column(String(200), nullable=True, index=True)
    container_url = Column(String(500), nullable=True)
    
    # Additional config (JSON) for CRM-specific settings
    # For Monday.com: workspace_id
    # For ClickUp: space_id, folder_id
    # For Jira: email, server_url
    # For Trello: api_token
    additional_config = Column(String, nullable=True)  # JSON string
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=True)
    
    # Relationships
    creator = relationship("User", foreign_keys=[created_by])

    def __repr__(self) -> str:
        return f"<CRMConfig(crm_type={self.crm_type}, container_id={self.container_id})>"

