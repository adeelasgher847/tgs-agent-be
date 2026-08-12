import pytest
from unittest.mock import patch, AsyncMock
import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.integrations.esp.client import EspClient
from app.integrations.esp.schemas import EspCustomerLookupRequest
from app.integrations.esp.service import (
    lookup_customer_service,
    extract_caller_phone,
    handle_trillet_webhook_service,
)

RAW_ESP_ITEM = {
    "CONTRACTID": 4754,
    "CONTRACTNUMBER": "HDP00695",
    "FIRSTNAME": "CLIFFORD",
    "LASTNAME": "HARRELL",
    "PHONE1": "(248) 625-0723",
    "EMAIL": None,
    "COVERAGENAME": "Enhanced Powertrain",
    "COVERAGETYPE": "Powertrain",
    "RETAILCOST": 3152.00,
    "SALEDATE": "/Date(1785992400000)/",
    "EXPIRATIONDATE": "/Date(1946350800000)/",
    "CLAIMNOTE": None,
}

REQ_OBJ = httpx.Request("GET", "https://expressserviceprotection.inlineadmin.com")


@pytest.mark.asyncio
async def test_client_lookup_by_contract_number():
    client = EspClient()
    mock_response = httpx.Response(200, json=[RAW_ESP_ITEM], request=REQ_OBJ)

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        res = await client.lookup_customer("HDP00695", "", "", "")

        assert len(res) == 1
        assert res[0]["CONTRACTNUMBER"] == "HDP00695"


def test_extract_caller_phone_candidates():
    assert extract_caller_phone({}, {"phone_number": "+12486250723"}) == "+12486250723"
    assert extract_caller_phone({"from": "+12486250723"}, {}) == "+12486250723"


@pytest.mark.asyncio
async def test_happyassist_is_test_connectivity_payload():
    test_payload = {
        "callId": "test-call-123",
        "isTest": True,
    }
    res = await handle_trillet_webhook_service(test_payload, {})
    assert res.variables == {}


def test_happyassist_webhook_router_endpoint():
    test_client = TestClient(app)

    test_payload = {
        "callId": "test-call-123",
        "isTest": True,
    }

    res = test_client.post("/api/v1/integrations/esp/happyassist-webhook", json=test_payload)
    assert res.status_code == 200
    assert res.json()["variables"] == {}
