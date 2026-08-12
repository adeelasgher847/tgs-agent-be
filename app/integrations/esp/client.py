import httpx
import logging
from typing import Any, List
from app.integrations.esp.config import esp_settings

logger = logging.getLogger(__name__)

class EspClientError(Exception):
    pass

def _format_param(value: str | None) -> str:
    if not value or not value.strip():
        return "''"
    return value.strip()

class EspClient:
    async def lookup_customer(self, contract_number: str, full_name: str, phone1: str, phone2: str) -> List[dict[str, Any]]:
        base_url = esp_settings.api_base_url.rstrip('/')
        query_id = esp_settings.query_id
        session_cookie = esp_settings.session_cookie

        # ESP requires empty search fields to be passed as literal single quotes ('').
        params = {
            "queryId": query_id,
            "ContractNumber": _format_param(contract_number),
            "FullName": _format_param(full_name),
            "Phone1": _format_param(phone1),
            "Phone2": _format_param(phone2)
        }
        
        cookies = {}
        if session_cookie:
            cookies["ASP.NET_SessionId"] = session_cookie

        url = f"{base_url}/services/administrationapi/00180000000E5bcNsePronuhZOjquEOdg/GetQueryData"
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params, cookies=cookies)
                response.raise_for_status()
                
                data = response.json()
                
                if not isinstance(data, list):
                    if data is None:
                        return []
                    if isinstance(data, dict):
                        return [data]
                    logger.error(f"ESP API returned unexpected non-list response: {type(data)}")
                    raise EspClientError("Invalid response format from ESP API.")
                
                return data

        except httpx.TimeoutException as e:
            logger.error(f"ESP API timeout: {e}")
            raise EspClientError("Request to ESP API timed out.")
        except httpx.HTTPStatusError as e:
            logger.error(f"ESP API HTTP error {e.response.status_code}: {e}")
            raise EspClientError(f"ESP API returned HTTP {e.response.status_code}.")
        except httpx.RequestError as e:
            logger.error(f"ESP API connection error: {e}")
            raise EspClientError("Failed to connect to ESP API.")
        except EspClientError:
            raise
        except Exception as e:
            logger.error(f"ESP API unexpected error: {e}")
            raise EspClientError("An unexpected error occurred while communicating with ESP.")

esp_client = EspClient()
