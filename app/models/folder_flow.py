from sqlalchemy import Column, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from app.db.base_class import Base


class FolderFlow(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    folder_id = Column(UUID(as_uuid=True), ForeignKey("folder.id", ondelete="CASCADE"), nullable=False, index=True)
    flow_id = Column(UUID(as_uuid=True), ForeignKey("callflow.id", ondelete="CASCADE"), nullable=False, index=True)

    folder = relationship("Folder", back_populates="folder_flows")
    call_flow = relationship("CallFlow")

    __table_args__ = (
        UniqueConstraint("folder_id", "flow_id", name="uq_folderflow_folder_flow"),
    )
