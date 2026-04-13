import enum
import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Enum, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base_class import Base


class ParseStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"


class UploadMode(str, enum.Enum):
    SINGLE = "SINGLE"
    BATCH = "BATCH"


class Resume(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    tenant_id = Column(
        UUID(as_uuid=True), ForeignKey("tenant.id"), index=True, nullable=False
    )
    original_filename = Column(String(512), nullable=False)
    content_type = Column(String(128), nullable=False)
    storage_path = Column(String(1024), nullable=False)
    status = Column(
        Enum(ParseStatus),
        default=ParseStatus.PENDING,
        nullable=False,
        index=True,
    )
    upload_mode = Column(
        Enum(UploadMode),
        default=UploadMode.SINGLE,
        nullable=False,
        index=True,
    )
    batch_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    job_description_id = Column(
        UUID(as_uuid=True), ForeignKey("jobdescription.id"), nullable=True, index=True
    )
    raw_text = Column(Text, nullable=True)
    parsed_json = Column(JSONB, nullable=True)
    warnings = Column(JSONB, nullable=True)
    parse_confidence = Column(Float, nullable=True)
    parse_source = Column(String(32), nullable=True)
    parser_version = Column(String(64), nullable=True)
    model_name = Column(String(128), nullable=True)
    provider = Column(String(64), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

