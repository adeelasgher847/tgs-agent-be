from typing import Tuple

from sqlalchemy.orm import Session

from app.core.logger import logger
from app.models.call_session import CallSession
from app.services.twilio_service import twilio_service


def get_twilio_credentials_for_call(db: Session, call_session: CallSession) -> Tuple[str, str]:
    """
    Get Twilio credentials for a call session.
    Priority: DB phone number credentials > Env credentials

    Returns:
        tuple: (account_sid, auth_token)
    """
    from app.models.phone_number import PhoneNumber
    from app.core.security import decrypt_api_key

    # Check if call was made with DB phone number
    if call_session.from_number:
        phone_number_obj = (
            db.query(PhoneNumber)
            .filter(
                PhoneNumber.phone_number == call_session.from_number,
                PhoneNumber.tenant_id == call_session.tenant_id,
                PhoneNumber.status == "active",
            )
            .first()
        )

        if (
            phone_number_obj
            and phone_number_obj.twilio_account_sid
            and phone_number_obj.twilio_auth_token
        ):
            # ✅ Use DB credentials (decrypt both)
            account_sid = decrypt_api_key(phone_number_obj.twilio_account_sid)
            auth_token = decrypt_api_key(phone_number_obj.twilio_auth_token)
            logger.info(
                "✅ Using DB credentials for recording (phone: %s)",
                call_session.from_number,
            )
            return account_sid, auth_token

    # ✅ Fallback to env credentials
    client = twilio_service.get_client()
    account_sid = client.username
    auth_token = client.password
    logger.info("✅ Using env credentials for recording")
    return account_sid, auth_token

