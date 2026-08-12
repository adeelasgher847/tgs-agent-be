import pytest
from unittest.mock import patch, AsyncMock
import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.integrations.esp.client import EspClient, EspClientError
from app.integrations.esp.schemas import EspCustomerLookupRequest
from app.integrations.esp.service import lookup_customer_service

# Sample raw ESP response item
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
    "CLAIMNOTE": None
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
        mock_get.assert_called_once()
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["ContractNumber"] == "HDP00695"
        assert kwargs["params"]["FullName"] == "''"
        assert kwargs["params"]["Phone1"] == "''"
        assert kwargs["params"]["Phone2"] == "''"


@pytest.mark.asyncio
async def test_client_lookup_by_full_name():
    client = EspClient()
    mock_response = httpx.Response(200, json=[RAW_ESP_ITEM], request=REQ_OBJ)
    
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        res = await client.lookup_customer("", "CLIFFORD HARRELL", "", "")
        
        assert len(res) == 1
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["ContractNumber"] == "''"
        assert kwargs["params"]["FullName"] == "CLIFFORD HARRELL"
        assert kwargs["params"]["Phone1"] == "''"
        assert kwargs["params"]["Phone2"] == "''"


@pytest.mark.asyncio
async def test_client_lookup_by_phone1():
    client = EspClient()
    mock_response = httpx.Response(200, json=[RAW_ESP_ITEM], request=REQ_OBJ)
    
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        res = await client.lookup_customer("", "", "(248) 625-0723", "")
        
        assert len(res) == 1
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["Phone1"] == "(248) 625-0723"


@pytest.mark.asyncio
async def test_client_lookup_by_phone2():
    client = EspClient()
    mock_response = httpx.Response(200, json=[RAW_ESP_ITEM], request=REQ_OBJ)
    
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        res = await client.lookup_customer("", "", "", "(248) 625-0724")
        
        assert len(res) == 1
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["Phone2"] == "(248) 625-0724"


@pytest.mark.asyncio
async def test_client_multiple_search_fields():
    client = EspClient()
    mock_response = httpx.Response(200, json=[RAW_ESP_ITEM], request=REQ_OBJ)
    
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        res = await client.lookup_customer("HDP00695", "CLIFFORD HARRELL", "(248) 625-0723", "")
        
        assert len(res) == 1
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["ContractNumber"] == "HDP00695"
        assert kwargs["params"]["FullName"] == "CLIFFORD HARRELL"
        assert kwargs["params"]["Phone1"] == "(248) 625-0723"
        assert kwargs["params"]["Phone2"] == "''"


@pytest.mark.asyncio
async def test_client_no_customer_found():
    client = EspClient()
    mock_response = httpx.Response(200, json=[], request=REQ_OBJ)
    
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        res = await client.lookup_customer("INVALID", "", "", "")
        assert res == []


@pytest.mark.asyncio
async def test_client_timeout():
    client = EspClient()
    
    with patch("httpx.AsyncClient.get", side_effect=httpx.TimeoutException("Timeout")):
        with pytest.raises(EspClientError) as exc_info:
            await client.lookup_customer("HDP00695", "", "", "")
        assert "timed out" in str(exc_info.value)


@pytest.mark.asyncio
async def test_client_http_error():
    client = EspClient()
    mock_response = httpx.Response(500, request=REQ_OBJ)
    
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        with pytest.raises(EspClientError) as exc_info:
            await client.lookup_customer("HDP00695", "", "", "")
        assert "500" in str(exc_info.value)


@pytest.mark.asyncio
async def test_client_invalid_response_format():
    client = EspClient()
    mock_response = httpx.Response(200, json=12345, request=REQ_OBJ)
    
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        with pytest.raises(EspClientError) as exc_info:
            await client.lookup_customer("HDP00695", "", "", "")
        assert "Invalid response format" in str(exc_info.value)


@pytest.mark.asyncio
async def test_service_multiple_records():
    request = EspCustomerLookupRequest(contract_number="HDP00695")
    raw_list = [RAW_ESP_ITEM, {**RAW_ESP_ITEM, "CONTRACTID": 4755, "CONTRACTNUMBER": "HDP00696"}]
    
    with patch("app.integrations.esp.service.esp_client.lookup_customer", new_callable=AsyncMock) as mock_lookup:
        mock_lookup.return_value = raw_list
        response = await lookup_customer_service(request)
        
        assert response.success is True
        assert len(response.customers) == 2
        assert response.customers[0].contract_id == 4754
        assert response.customers[1].contract_id == 4755


def test_router_unauthenticated_access_success():
    """Verify endpoint is public and requires no auth headers."""
    test_client = TestClient(app)
    
    with patch("app.integrations.esp.service.esp_client.lookup_customer", new_callable=AsyncMock) as mock_lookup:
        mock_lookup.return_value = [RAW_ESP_ITEM]
        
        # No Authorization or x-api-key headers provided
        response = test_client.post("/api/v1/integrations/esp/customer-lookup", json={
            "contract_number": "HDP00695",
            "full_name": "",
            "phone1": "",
            "phone2": ""
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["customers"]) == 1
        assert data["customers"][0]["contract_number"] == "HDP00695"
