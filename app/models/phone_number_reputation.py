from __future__ import annotations

import uuid

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base_class import Base


class PhoneNumberReputation(Base):
    """Carrier spam/reputation check result for an outbound phone number."""

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    phone_number_id = Column(
        UUID(as_uuid=True),
        ForeignKey("phonenumber.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    reputation_score = Column(Integer, nullable=False, default=100, server_default="100")
    spam_flagged = Column(Boolean, nullable=False, default=False, server_default="false")
    last_checked_at = Column(DateTime(timezone=True), nullable=True)
    # e.g. 'first_orion', 'hiya', 'mock'
    checked_by = Column(String, nullable=True)
    flagged_reason = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    phone_number = relationship("PhoneNumber", back_populates="reputation")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<PhoneNumberReputation phone_number_id={self.phone_number_id} "
            f"score={self.reputation_score} flagged={self.spam_flagged}>"
        )
