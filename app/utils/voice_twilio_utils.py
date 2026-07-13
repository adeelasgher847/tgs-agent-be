from typing import Optional, Tuple

from sqlalchemy.orm import Session

from app.core.logger import logger
from app.models.call_session import CallSession


def get_twilio_credentials_for_call(db: Session, call_session: CallSession) -> Tuple[str, str]:
    """
    Get Twilio credentials for a call session from DB phone number mapping only.

    Returns:
        tuple: (account_sid, auth_token)
    """
    from app.models.phone_number import PhoneNumber
    from app.core.security import decrypt_api_key

    candidate_numbers = [
        call_session.assistant_phone_number,
        call_session.to_number,
        call_session.from_number,
    ]
    candidate_numbers = [n for n in candidate_numbers if n]

    if not candidate_numbers:
        raise ValueError("No phone number available on call session to resolve Twilio credentials")

    phone_number_obj = (
        db.query(PhoneNumber)
        .filter(
            PhoneNumber.phone_number.in_(candidate_numbers),
            PhoneNumber.tenant_id == call_session.tenant_id,
            PhoneNumber.status == "active",
        )
        .first()
    )

    if (
        not phone_number_obj
        or not phone_number_obj.twilio_account_sid
        or not phone_number_obj.twilio_auth_token
    ):
        raise ValueError(
            f"Twilio credentials not found in DB for call session {call_session.id}"
        )

    account_sid = decrypt_api_key(phone_number_obj.twilio_account_sid)
    auth_token = decrypt_api_key(phone_number_obj.twilio_auth_token)
    logger.info(
        "✅ Using DB Twilio credentials for call session %s (number: %s)",
        call_session.id,
        phone_number_obj.phone_number,
    )
    return account_sid, auth_token


def twilio_caller_id_for_transfer_dial(call_session: CallSession) -> Optional[str]:
    """
    Caller ID for outbound <Dial> / REST calls must be a Twilio or verified number — never the customer's E.164.

    - Inbound: assistant_phone_number or to_number (the Twilio number that was dialed).
    - Outbound: assistant_phone_number or from_number (the Twilio line used to place the call).
    """
    if call_session.assistant_phone_number:
        return call_session.assistant_phone_number.strip()
    ctype = (call_session.call_type or "").lower()
    if ctype == "outbound" and call_session.from_number:
        return call_session.from_number.strip()
    if ctype == "inbound" and call_session.to_number:
        return call_session.to_number.strip()
    return None

