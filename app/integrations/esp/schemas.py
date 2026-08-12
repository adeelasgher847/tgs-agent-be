from typing import List, Optional
from pydantic import BaseModel, Field

class EspCustomerLookupRequest(BaseModel):
    contract_number: str = Field(default="", description="The contract number, e.g. HDP00695")
    full_name: str = Field(default="", description="The customer's full name")
    phone1: str = Field(default="", description="The customer's primary phone")
    phone2: str = Field(default="", description="The customer's secondary phone")

class EspCustomerRecord(BaseModel):
    contract_id: Optional[int] = Field(None, validation_alias="CONTRACTID")
    contract_number: Optional[str] = Field(None, validation_alias="CONTRACTNUMBER")
    first_name: Optional[str] = Field(None, validation_alias="FIRSTNAME")
    last_name: Optional[str] = Field(None, validation_alias="LASTNAME")
    phone1: Optional[str] = Field(None, validation_alias="PHONE1")
    email: Optional[str] = Field(None, validation_alias="EMAIL")
    coverage_name: Optional[str] = Field(None, validation_alias="COVERAGENAME")
    coverage_type: Optional[str] = Field(None, validation_alias="COVERAGETYPE")
    retail_cost: Optional[float] = Field(None, validation_alias="RETAILCOST")
    sale_date: Optional[str] = Field(None, validation_alias="SALEDATE")
    expiration_date: Optional[str] = Field(None, validation_alias="EXPIRATIONDATE")
    claim_note: Optional[str] = Field(None, validation_alias="CLAIMNOTE")

    model_config = {
        "populate_by_name": True,
    }

class EspCustomerLookupResponse(BaseModel):
    success: bool
    customers: List[EspCustomerRecord] = Field(default_factory=list)
    error: Optional[str] = None
