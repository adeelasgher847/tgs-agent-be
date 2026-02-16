"""
Vicidial Dialer Service
Handles Vicidial call operations using NON-AGENT API
"""

import requests
import urllib3
from typing import Dict, Optional, Any
from urllib.parse import urlencode
from app.services.base_dialer_service import BaseDialerService
from app.core.config import settings
from app.core.logger import logger

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class VicidialDialerService(BaseDialerService):
    """Vicidial implementation of BaseDialerService"""
    
    def __init__(self):
        self.base_url = settings.VICIDIAL_BASE_URL
        self.api_user = settings.VICIDIAL_API_USER
        self.api_pass = settings.VICIDIAL_API_PASS
    
    def _make_api_call(self, function: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a NON-AGENT API call to Vicidial
        
        Args:
            function: API function name
            params: Additional parameters
            
        Returns:
            Dict with API response
        """
        url = f"{self.base_url}/non_agent_api.php"
        
        query_params = {
            "function": function,
            "user": self.api_user,
            "pass": self.api_pass,
            "source": "backend"
        }
        
        if params:
            query_params.update(params)
        
        try:
            response = requests.get(url, params=query_params, verify=False, timeout=15)
            response.raise_for_status()
            
            # Parse response (Vicidial returns pipe-delimited or JSON)
            text = response.text.strip()
            
            # Check for errors
            if "ERROR" in text.upper():
                logger.error(f"❌ Vicidial API error: {text}")
                return {"success": False, "error": text}
            
            return {"success": True, "data": text, "raw_response": response.text}
        except Exception as e:
            logger.error(f"❌ Error calling Vicidial API {function}: {e}")
            return {"success": False, "error": str(e)}
    
    def initiate_call(
        self,
        to_number: str,
        from_number: str,
        webhook_url: str,
        status_callback_url: str,
        call_session_id: str,
        campaign_id: str,
        list_id: str,
        phone_code: str = "1",
        phone_number: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Initiate a call using Vicidial by adding lead to hopper
        
        Args:
            to_number: Destination phone number (without +)
            from_number: Source phone number (not used directly, uses CID group)
            webhook_url: Webhook URL (for future FastAGI integration)
            status_callback_url: Status callback URL
            call_session_id: Call session ID (stored in vendor_lead_code)
            campaign_id: Vicidial campaign ID
            list_id: Vicidial list ID
            phone_code: Country code (default: 1)
            phone_number: Phone number (will use to_number if not provided)
            **kwargs: Additional parameters (caller_id_number, etc.)
            
        Returns:
            Dict with call information (call_id, status, etc.)
        """
        try:
            # Use phone_number if provided, otherwise use to_number
            phone = phone_number if phone_number else to_number.replace("+", "")
            
            # Prepare lead data (Vicidial NON-AGENT API add_lead)
            # add_to_hopper=Y puts the lead directly in the dial hopper so the dialer can pick it up immediately
            # See: https://www.vicidial.org/docs/NON-AGENT_API.txt
            lead_params = {
                "function": "add_lead",
                "campaign_id": campaign_id,
                "phone_number": phone,
                "phone_code": phone_code,
                "vendor_lead_code": call_session_id,  # Store our call_session_id
                "source": "backend_api",
                "add_to_hopper": "Y",  # Put lead in hopper immediately so dialer can dial (NON-AGENT API)
            }
            
            # Add optional fields
            if kwargs.get("caller_id_number"):
                lead_params["caller_id_number"] = kwargs["caller_id_number"]
            
            if kwargs.get("first_name"):
                lead_params["first_name"] = kwargs["first_name"]
            if kwargs.get("last_name"):
                lead_params["last_name"] = kwargs["last_name"]
            
            result = self._make_api_call("add_lead", lead_params)
            
            if not result.get("success"):
                raise Exception(f"Failed to add lead: {result.get('error', 'Unknown error')}")
            
            # Extract lead_id from response (Vicidial returns: SUCCESS|lead_id|...)
            response_data = result.get("data", "")
            lead_id = None
            if "SUCCESS" in response_data:
                parts = response_data.split("|")
                if len(parts) > 1:
                    lead_id = parts[1].strip()
            
            logger.info(f"✅ Vicidial lead added (with add_to_hopper=Y): lead_id={lead_id}, campaign={campaign_id}, phone={phone}")
            
            return {
                "call_id": lead_id or call_session_id,
                "lead_id": lead_id,
                "status": "queued",
                "dialer_type": "vicidial",
                "campaign_id": campaign_id,
                "list_id": list_id,
                "from_number": from_number,
                "to_number": to_number,
                "vendor_lead_code": call_session_id,
            }
        except Exception as e:
            logger.error(f"❌ Error initiating Vicidial call: {e}")
            raise
    
    def end_call(self, call_id: str, **kwargs) -> bool:
        """
        End a Vicidial call
        
        Note: Vicidial doesn't have direct call termination API
        This would require AMI (Asterisk Manager Interface) or FastAGI
        
        Args:
            call_id: Vicidial call ID or lead ID
            **kwargs: Additional parameters
            
        Returns:
            bool: True if call ended successfully
        """
        try:
            # For now, we can't directly end calls via NON-AGENT API
            # This would require AMI or FastAGI integration
            logger.warning(f"⚠️ Direct call termination not available via NON-AGENT API for call {call_id}")
            logger.info(f"💡 Call termination will be handled via FastAGI bridge (Part 3)")
            return False
        except Exception as e:
            logger.error(f"❌ Error ending Vicidial call {call_id}: {e}")
            return False
    
    def get_call_status(self, call_id: str, **kwargs) -> Dict[str, Any]:
        """
        Get Vicidial call status using callid_info API
        
        Args:
            call_id: Vicidial call ID
            **kwargs: Additional parameters
            
        Returns:
            Dict with call status information
        """
        try:
            result = self._make_api_call("callid_info", {"call_id": call_id})
            
            if not result.get("success"):
                return {"call_id": call_id, "status": "unknown", "error": result.get("error")}
            
            # Parse response (pipe-delimited format)
            data = result.get("data", "")
            # Format: call_id|status|duration|...
            
            return {
                "call_id": call_id,
                "status": "active",  # Will be parsed from response
                "raw_data": data
            }
        except Exception as e:
            logger.error(f"❌ Error getting Vicidial call status {call_id}: {e}")
            return {"call_id": call_id, "status": "unknown", "error": str(e)}
    
    def get_recording_url(self, call_id: str, **kwargs) -> Optional[str]:
        """
        Get Vicidial call recording URL using recording_lookup API
        
        Args:
            call_id: Vicidial call ID
            **kwargs: Additional parameters
            
        Returns:
            Recording URL or None
        """
        try:
            result = self._make_api_call("recording_lookup", {"call_id": call_id})
            
            if not result.get("success"):
                return None
            
            # Parse response to extract recording URL
            data = result.get("data", "")
            # Format varies, need to parse based on Vicidial response format
            
            # For now, return None - will be implemented based on actual response format
            logger.info(f"📹 Recording lookup for call {call_id}: {data}")
            return None
        except Exception as e:
            logger.error(f"❌ Error getting Vicidial recording URL for call {call_id}: {e}")
            return None
