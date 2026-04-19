"""
WATI (WhatsApp) API client — template messages for staff appointment prompts.
Docs: https://docs.wati.io/reference/post_api-v1-sendtemplatemessage
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.core.config import settings
from app.core.logger import logger


def normalize_whatsapp_number_e164(phone: Optional[str]) -> Optional[str]:
    """
    Return a best-effort E.164-style +digits string, US-first.

    - 10 digits → +1 (NANP) for US-heavy deployments.
    - 11 digits starting with 1 → +<digits> (US with country code).
    - Long strings with a spurious leading 0 before a full country code (e.g. 0923…)
      strip leading zeros so +092… is avoided.
    Non-US national formats (e.g. 03xx…) without country code are not fully normalized.
    """
    if not phone or not str(phone).strip():
        return None
    digits = re.sub(r"\D", "", str(phone).strip())
    if not digits:
        return None
    if len(digits) >= 12 and digits.startswith("0"):
        digits = digits.lstrip("0") or digits

    if len(digits) == 10:
        return f"+1{digits}"
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    return f"+{digits}"


def extract_wati_inbound_text_and_sender(payload: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Best-effort parse for WATI inbound webhooks (payload shapes vary by event/version).
    Returns (message_text, sender_phone_raw).
    """
    if payload is None:
        return None, None
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if not isinstance(payload, dict):
        return None, None

    def _dig_text(obj: Any) -> Optional[str]:
        if not isinstance(obj, dict):
            return None
        for k in ("text", "messageText", "body", "message"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    text = _dig_text(payload)
    wm = payload.get("whatsappMessage") or payload.get("message") or payload.get("data")
    if not text and isinstance(wm, dict):
        text = _dig_text(wm)
        nested = wm.get("whatsappMessage")
        if not text and isinstance(nested, dict):
            text = _dig_text(nested)

    sender = (
        payload.get("waId")
        or payload.get("whatsappNumber")
        or payload.get("phone")
        or payload.get("senderPhone")
    )
    if not sender and isinstance(wm, dict):
        sender = wm.get("owner")  # sometimes bool — skip
        if not isinstance(sender, str):
            sender = wm.get("chatId") or wm.get("from")
    if not isinstance(sender, str):
        sender = None

    return text, sender


def normalize_whatsapp_number_wati_query(phone: Optional[str]) -> Optional[str]:
    """
    WATI sendTemplateMessage query param: country code + national number, no + sign.
    Example: 85264318721
    """
    e164 = normalize_whatsapp_number_e164(phone)
    if not e164:
        return None
    return e164.lstrip("+")


class WatiService:
    def _base_url(self) -> str:
        base = (settings.WATI_API_BASE_URL or "").strip().rstrip("/")
        return base

    def is_configured(self) -> bool:
        return bool(
            settings.WATI_ENABLED
            and self._base_url()
            and (settings.WATI_ACCESS_TOKEN or "").strip()
            and (settings.WATI_TEMPLATE_NAME or "").strip()
            and (settings.WATI_CHANNEL_NUMBER or "").strip()
        )

    def send_template_message(
        self,
        *,
        whatsapp_number: str,
        template_name: str,
        parameters: List[Dict[str, str]],
        broadcast_name: str,
        channel_number: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        POST /api/v1/sendTemplateMessage?whatsappNumber=...
        Raises RuntimeError on missing config or non-success HTTP.
        """
        if not settings.WATI_ENABLED:
            raise RuntimeError("WATI is disabled (WATI_ENABLED=false)")
        base = self._base_url()
        token = (settings.WATI_ACCESS_TOKEN or "").strip()
        if not base or not token:
            raise RuntimeError("WATI_API_BASE_URL and WATI_ACCESS_TOKEN are required")
        ch = (channel_number or settings.WATI_CHANNEL_NUMBER or "").strip()
        if not ch:
            raise RuntimeError("WATI_CHANNEL_NUMBER is required for sendTemplateMessage")

        wa_query = normalize_whatsapp_number_wati_query(whatsapp_number)
        if not wa_query:
            raise RuntimeError("Invalid staff WhatsApp number")

        url = f"{base}/api/v1/sendTemplateMessage"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        body = {
            "template_name": template_name,
            "broadcast_name": broadcast_name[:80],
            "channel_number": ch,
            "parameters": parameters,
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                url,
                params={"whatsappNumber": wa_query},
                headers=headers,
                json=body,
            )
        try:
            data = resp.json()
        except Exception:
            data = {"raw": resp.text[:500]}
        if resp.status_code >= 400:
            logger.error(
                "WATI sendTemplateMessage HTTP %s: %s",
                resp.status_code,
                data,
            )
            raise RuntimeError(f"WATI sendTemplateMessage failed: HTTP {resp.status_code}")
        if not data.get("result", True):
            logger.error("WATI sendTemplateMessage result=false: %s", data)
            raise RuntimeError("WATI sendTemplateMessage failed: result=false")
        logger.info(
            "WATI template sent template=%s to=***%s",
            template_name,
            wa_query[-4:],
        )
        return data

    def send_staff_appointment_prompt(
        self,
        *,
        staff_whatsapp_e164: str,
        customer_name: str,
        time_line: str,
        ack_code: str,
    ) -> Dict[str, Any]:
        """
        Send Meta-approved template. Default parameter names match common single-body templates;
        override via env if your template uses different placeholder names.
        """
        template = (settings.WATI_TEMPLATE_NAME or "").strip()
        instruction = (
            f"Reply YES or CONFIRM to approve. Booking code: {ack_code}"
        )
        raw_names = (settings.WATI_TEMPLATE_PARAM_NAMES or "").strip()
        names = [n.strip() for n in raw_names.split(",") if n.strip()]
        if not names:
            names = ["customer_name", "time_line", "ack_code", "instructions"]
        values = [
            customer_name[:200],
            time_line[:500],
            ack_code,
            instruction[:500],
        ]
        parameters = []
        for i, name in enumerate(names):
            val = values[i] if i < len(values) else ""
            parameters.append({"name": name, "value": val})
        broadcast = f"appt_staff_{ack_code}"
        return self.send_template_message(
            whatsapp_number=staff_whatsapp_e164,
            template_name=template,
            parameters=parameters,
            broadcast_name=broadcast,
        )


wati_service = WatiService()
