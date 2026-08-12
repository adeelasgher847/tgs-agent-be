import httpx
import logging
from typing import Any, List
from app.integrations.esp.config import esp_settings

logger = logging.getLogger(__name__)

class EspClientError(Exception):
    pass

class EspClient:
    def __init__(self):
        self.base_url = esp_settings.api_base_url.rstrip('/')
        self.query_id = esp_settings.query_id
        self.session_cookie = esp_settings.session_cookie

    async def lookup_customer(self, contract_number: str, full_name: str, phone1: str, phone2: str) -> List[dict[str, Any]]:
        # ESP requires empty search fields to be passed as an empty string.
        params = {
            "queryId": self.query_id,
            "ContractNumber": contract_number or "",
            "FullName": full_name or "",
            "Phone1": phone1 or "",
            "Phone2": phone2 or ""
        }
        
        cookies = {}
        if self.session_cookie:
            cookies["ASP.NET_SessionId"] = self.session_cookie

        url = f"{self.base_url}/services/administrationapi/00180000000E5bcNsePronuhZOjquEOdg/GetQueryData"
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params, cookies=cookies)
                response.raise_for_status()
                
                data = response.json()
                
                if not isinstance(data, list):
                    # Sometimes APIs return a single object or null, normalize it to list
                    if data is None:
                        return []
                    # if it's a dict, we might wrap it in a list
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
