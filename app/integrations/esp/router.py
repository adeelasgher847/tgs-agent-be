from fastapi import APIRouter, Request
from app.integrations.esp.schemas import (
    EspCustomerLookupRequest,
    EspCustomerLookupResponse,
    TrilletWebhookResponse,
)
from app.integrations.esp.service import (
    handle_trillet_webhook_service,
    lookup_customer_service,
)

router = APIRouter()

@router.post(
    "/customer-lookup",
    response_model=EspCustomerLookupResponse,
    summary="ESP Customer Lookup",
    description="Query Express Service Protection (ESP) for customer records."
)
async def esp_customer_lookup(
    request: EspCustomerLookupRequest
) -> EspCustomerLookupResponse:
    return await lookup_customer_service(request)


@router.post(
    "/happyassist-webhook",
    response_model=TrilletWebhookResponse,
    summary="HappyAssist Webhook Adapter for ESP Customer Lookup",
    description=(
        "Receives HappyAssist/Trillet pre-inbound call webhooks. Safely handles connectivity test payloads (isTest=True) "
        "and queries ESP using caller phone number to inject context variables into the AI prompt."
    )
)
@router.post(
    "/trillet-webhook",
    response_model=TrilletWebhookResponse,
    include_in_schema=False,
)
async def happyassist_webhook(
    request: Request
) -> TrilletWebhookResponse:
    try:
        raw_payload = await request.json()
    except Exception:
        raw_payload = {}
    query_params = dict(request.query_params)
    return await handle_trillet_webhook_service(raw_payload, query_params)
