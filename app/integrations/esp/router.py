from fastapi import APIRouter
from app.integrations.esp.schemas import EspCustomerLookupRequest, EspCustomerLookupResponse
from app.integrations.esp.service import lookup_customer_service

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
