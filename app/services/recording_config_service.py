"""
Recording Config Service — resolves recording_enabled for a call session.

Looks up the NumberConfiguration for the phone number associated with a
CallSession and returns whether recording is enabled for that number.
"""

from __future__ import annotations


from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logger import logger
from app.models.call_session import CallSession
from app.models.phone_number import NumberConfiguration, PhoneNumber


def get_recording_enabled_for_call(db: Session, call_session: CallSession) -> bool:
    """
    Return True if recording is enabled for this call.

    Resolution order:
    1. If call_session.call_flow_id is present, resolve CallFlow.recording_enabled.
    2. For web calls (call_type == "web") without flow, fall back to VOICE_BROWSER_DEMO_RECORDING_ENABLED.
    3. Twilio calls (inbound/outbound) resolution order via NumberConfiguration:
       - call_session.assistant_phone_number (set on outbound and inbound calls)
       - call_session.to_number (fallback for inbound — the number the caller dialled)
       - call_session.from_number (last resort for outbound)
       Returns False if no NumberConfiguration is found (safe default).
    """
    if call_session.call_flow_id:
        try:
            from app.models.call_flow import CallFlow

            flow = db.execute(
                select(CallFlow).where(
                    CallFlow.id == call_session.call_flow_id,
                    CallFlow.tenant_id == call_session.tenant_id,
                    CallFlow.is_deleted.is_(False),
                )
            ).scalar_one_or_none()
            if flow is not None:
                return bool(
                    flow.recording_enabled
                    if flow.recording_enabled is not None
                    else True
                )
        except Exception as exc:
            logger.error(
                "recording_config: flow lookup failed due to DB error for session %s (flow=%s): %s",
                call_session.id,
                call_session.call_flow_id,
                exc,
                exc_info=True,
            )
            return False

    if (call_session.call_type or "").lower() == "web":
        return bool(settings.VOICE_BROWSER_DEMO_RECORDING_ENABLED)

    phone_number_str = _resolve_phone_number(call_session)
    if not phone_number_str:
        return False

    try:
        stmt = (
            select(NumberConfiguration)
            .join(PhoneNumber, NumberConfiguration.phone_number_id == PhoneNumber.id)
            .where(
                PhoneNumber.phone_number == phone_number_str,
                PhoneNumber.tenant_id == call_session.tenant_id,
            )
        )
        config = db.execute(stmt).scalar_one_or_none()
        if config is None:
            return False
        return bool(config.recording_enabled)
    except Exception as exc:
        logger.warning(
            "recording_config: lookup failed for session %s: %s",
            call_session.id,
            exc,
        )
        return False


def _resolve_phone_number(call_session: CallSession) -> str | None:
    for field in ("assistant_phone_number", "to_number", "from_number"):
        val = getattr(call_session, field, None)
        if val and str(val).strip():
            return str(val).strip()
    return None
