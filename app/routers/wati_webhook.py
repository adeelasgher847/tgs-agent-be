"""
WATI (WhatsApp) inbound webhooks — staff replies to confirm appointments.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.config import settings
from app.core.logger import logger
from app.services.calendar_service import calendar_service
from app.services.wati_service import extract_wati_inbound_text_and_sender

router = APIRouter()


@router.post("/wati/inbound", include_in_schema=False)
async def wati_inbound_webhook(
    request: Request,
    db: Session = Depends(get_db),
    secret: str | None = Query(
        None,
        description="When WATI_WEBHOOK_SECRET is set: must match (or use X-Wati-Webhook-Secret header).",
    ),
):
    # If a dedicated secret is configured, that alone authorizes the webhook (no global flag needed).
    expected = (settings.WATI_WEBHOOK_SECRET or "").strip()
    if expected:
        hdr = (request.headers.get("X-Wati-Webhook-Secret") or "").strip()
        q = (secret or "").strip()
        if hdr != expected and q != expected:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid webhook secret.",
            )
    elif not settings.ALLOW_UNAUTHENTICATED_WEBHOOKS:
        logger.warning(
            "WATI webhook rejected: WATI_WEBHOOK_SECRET unset and ALLOW_UNAUTHENTICATED_WEBHOOKS is false"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Unauthenticated webhooks are disabled. Set WATI_WEBHOOK_SECRET "
                "(recommended) or ALLOW_UNAUTHENTICATED_WEBHOOKS=true."
            ),
        )

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Expected JSON body.",
        ) from None

    text, sender = extract_wati_inbound_text_and_sender(body)
    if not text:
        return {"ok": True, "ignored": True, "reason": "no_text"}

    appt = calendar_service.try_confirm_appointment_from_wati_reply(
        db,
        inbound_text=text,
        sender_phone=sender,
    )
    if appt:
        logger.info(
            "WATI webhook confirmed appointment=%s tenant=%s",
            appt.id,
            appt.tenant_id,
        )
        return {"ok": True, "appointment_id": str(appt.id), "confirmed": True}
    return {"ok": True, "confirmed": False}
