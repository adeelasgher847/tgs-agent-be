from typing import Optional, Dict, Any

from sqlalchemy.orm import Session

from app.core.logger import logger
from app.models.call_session import CallSession
from app.services.transcript_service import transcript_service


async def add_to_transcript(
    call_session: CallSession,
    role: str,
    message: str,
    db: Session,
    message_type: str = "speech",
    agent_id: Optional[str] = None,
    user_id: Optional[str] = None,
    confidence: Optional[float] = None,
    duration: Optional[float] = None,
    response_time: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Add a message to the transcript using the new transcript service.

    This is a direct refactor of `_add_to_transcript` from `voice.py`
    with behavior preserved.
    """
    logger.debug("📝 Adding to transcript: %s - %s...", role, message[:50])

    try:
        transcript_message = await transcript_service.add_and_broadcast_message(
            db=db,
            call_session_id=call_session.id,
            role=role,
            message=message,
            message_type=message_type,
            agent_id=agent_id,
            user_id=user_id,
            confidence=confidence,
            duration=duration,
            response_time=response_time,
            metadata=metadata,
        )

        logger.debug(
            "✅ Added transcript message %s for session %s",
            transcript_message.id,
            call_session.id,
        )

        # Also update the legacy call_transcript field for backward compatibility
        conversation = transcript_service.get_conversation_array(db, call_session.id)
        call_session.call_transcript = conversation
        db.commit()

        return transcript_message

    except Exception as e:
        logger.error("❌ Failed to add transcript message: %s", e, exc_info=True)
        raise


def get_conversation_state(call_session: CallSession):
    """Helper function to get conversation state."""
    if not call_session.call_metadata:
        call_session.call_metadata = {}
    if "conversation_state" not in call_session.call_metadata:
        call_session.call_metadata["conversation_state"] = {}
    return call_session.call_metadata["conversation_state"]


def update_conversation_state(call_session: CallSession, key: str, value):
    """Helper function to update conversation state."""
    state = get_conversation_state(call_session)
    state[key] = value
    call_session.call_metadata["conversation_state"] = state

