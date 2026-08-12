from app.integrations.esp.client import esp_client, EspClientError
from app.integrations.esp.schemas import EspCustomerLookupRequest, EspCustomerLookupResponse, EspCustomerRecord

async def lookup_customer_service(request: EspCustomerLookupRequest) -> EspCustomerLookupResponse:
    try:
        raw_data = await esp_client.lookup_customer(
            contract_number=request.contract_number,
            full_name=request.full_name,
            phone1=request.phone1,
            phone2=request.phone2
        )
        
        customers = []
        for item in raw_data:
            # Pydantic will populate fields correctly based on the aliases we set
            try:
                record = EspCustomerRecord.model_validate(item)
                customers.append(record)
            except Exception as e:
                # If a single record fails validation, we log it and continue
                import logging
                logging.getLogger(__name__).warning(f"Failed to parse ESP customer record: {e}")
        
        return EspCustomerLookupResponse(
            success=True,
            customers=customers,
            error=None
        )
        
    except EspClientError as e:
        return EspCustomerLookupResponse(
            success=False,
            customers=[],
            error=str(e)
        )
    except Exception as e:
        return EspCustomerLookupResponse(
            success=False,
            customers=[],
            error="An internal error occurred."
        )
