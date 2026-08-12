import logging
from typing import Any, Dict, Optional
from app.integrations.esp.client import esp_client, EspClientError
from app.integrations.esp.config import esp_settings
from app.integrations.esp.schemas import (
    EspCustomerLookupRequest,
    EspCustomerLookupResponse,
    EspCustomerRecord,
    TrilletWebhookResponse,
)

logger = logging.getLogger(__name__)


def extract_caller_phone(payload: Dict[str, Any], query_params: Dict[str, str]) -> Optional[str]:
    """Extract caller phone number from Trillet query parameters or body payload."""

    # 1. Check Query Parameters (Trillet automatically sends ?phone_number=... or query params)
    for qk in ["phone_number", "phoneNumber", "from", "caller_number", "phone"]:
        val = query_params.get(qk)
        if val and isinstance(val, str) and val.strip():
            return val.strip()

    if not isinstance(payload, dict):
        return None

    # 2. Configured field override (supports dot-notation e.g. "data.callerNumber")
    custom_field = getattr(esp_settings, "trillet_phone_field", "").strip()
    if custom_field:
        val: Any = payload
        for part in custom_field.split("."):
            if isinstance(val, dict):
                val = val.get(part)
            else:
                val = None
                break
        if isinstance(val, str) and val.strip():
            return val.strip()

    # 3. Standard candidate body paths
    data_obj = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    customer_obj = payload.get("customer") if isinstance(payload.get("customer"), dict) else {}

    candidates = [
        payload.get("phone_number"),
        payload.get("from"),
        payload.get("callerNumber"),
        payload.get("caller_number"),
        payload.get("phone"),
        payload.get("phoneNumber"),
        data_obj.get("phone_number"),
        data_obj.get("from"),
        data_obj.get("callerNumber"),
        data_obj.get("caller_number"),
        data_obj.get("phone"),
        customer_obj.get("phone_number"),
        customer_obj.get("phone"),
        customer_obj.get("phoneNumber"),
    ]

    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip()

    return None


async def lookup_customer_service(request: EspCustomerLookupRequest) -> EspCustomerLookupResponse:
    try:
        raw_data = await esp_client.lookup_customer(
            contract_number=request.contract_number,
            full_name=request.full_name,
            phone1=request.phone1,
            phone2=request.phone2,
        )

        customers = []
        for item in raw_data:
            try:
                record = EspCustomerRecord.model_validate(item)
                customers.append(record)
            except Exception as e:
                logger.warning(f"Failed to parse ESP customer record: {e}")

        return EspCustomerLookupResponse(
            success=True,
            customers=customers,
            error=None,
        )

    except EspClientError as e:
        return EspCustomerLookupResponse(
            success=False,
            customers=[],
            error=str(e),
        )
    except Exception:
        return EspCustomerLookupResponse(
            success=False,
            customers=[],
            error="An internal error occurred.",
        )


async def handle_trillet_webhook_service(
    payload: Dict[str, Any], query_params: Dict[str, str]
) -> TrilletWebhookResponse:
    if not isinstance(payload, dict):
        payload = {}

    # 1. Detect test connectivity payload
    data_obj = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    is_test = payload.get("isTest") is True or data_obj.get("event") == "test"

    if is_test:
        logger.info("Received Trillet webhook test payload (isTest=True). Returning empty variables.")
        return TrilletWebhookResponse(variables={})

    # 2. Extract caller phone number from Query Params or JSON Body
    phone_number = extract_caller_phone(payload, query_params)
    if not phone_number:
        logger.warning("Trillet webhook request missing caller phone number.")
        return TrilletWebhookResponse(variables={})

    # 3. Perform ESP customer lookup using caller phone as phone1
    lookup_request = EspCustomerLookupRequest(phone1=phone_number)
    lookup_response = await lookup_customer_service(lookup_request)

    if not lookup_response.success or not lookup_response.customers:
        return TrilletWebhookResponse(variables={})

    # 4. Format customer record into Trillet documented response schema
    c = lookup_response.customers[0]
    customer_name = f"{c.first_name or ''} {c.last_name or ''}".strip()

    variables = {
        "customer_name": str(customer_name),
        "contract_number": str(c.contract_number or ""),
        "coverage_name": str(c.coverage_name or ""),
        "coverage_type": str(c.coverage_type or ""),
        "retail_cost": str(c.retail_cost) if c.retail_cost is not None else "",
        "expiration_date": str(c.expiration_date or ""),
    }

    return TrilletWebhookResponse(variables=variables)
