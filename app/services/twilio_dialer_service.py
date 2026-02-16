"""
Twilio Dialer Service
Wrapper around TwilioService implementing BaseDialerService interface
"""

from typing import Dict, Optional, Any
from app.services.base_dialer_service import BaseDialerService
from app.services.twilio_service import twilio_service
from app.core.logger import logger


class TwilioDialerService(BaseDialerService):
    """Twilio implementation of BaseDialerService"""
    
    def initiate_call(
        self,
        to_number: str,
        from_number: str,
        webhook_url: str,
        status_callback_url: str,
        call_session_id: str,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Initiate a call using Twilio
        
        Args:
            to_number: Destination phone number
            from_number: Source phone number
            webhook_url: Webhook URL for call events
            status_callback_url: Status callback URL
            call_session_id: Call session ID
            account_sid: Optional custom Twilio Account SID
            auth_token: Optional custom Twilio Auth Token
            **kwargs: Additional parameters
            
        Returns:
            Dict with call information (call_id, status, etc.)
        """
        try:
            if account_sid and auth_token:
                # Use custom credentials
                call = twilio_service.make_call_with_credentials(
                    to_number=to_number,
                    from_number=from_number,
                    webhook_url=webhook_url,
                    status_callback_url=status_callback_url,
                    account_sid=account_sid,
                    auth_token=auth_token,
                    record=kwargs.get('record', True)
                )
            else:
                # Use default credentials from env
                call = twilio_service.make_call(
                    to_number=to_number,
                    from_number=from_number,
                    webhook_url=webhook_url,
                    status_callback_url=status_callback_url,
                    record=kwargs.get('record', True)
                )
            
            logger.info(f"✅ Twilio call initiated: SID={call.sid}, To={to_number}, From={from_number}")
            
            return {
                "call_id": call.sid,
                "status": call.status,
                "dialer_type": "twilio",
                "from_number": from_number,
                "to_number": to_number
            }
        except Exception as e:
            logger.error(f"❌ Error initiating Twilio call: {e}")
            raise
    
    def end_call(self, call_id: str, account_sid: Optional[str] = None, auth_token: Optional[str] = None, **kwargs) -> bool:
        """
        End a Twilio call
        
        Args:
            call_id: Twilio Call SID
            account_sid: Optional custom Twilio Account SID
            auth_token: Optional custom Twilio Auth Token
            **kwargs: Additional parameters
            
        Returns:
            bool: True if call ended successfully
        """
        try:
            return twilio_service.end_call(call_id)
        except Exception as e:
            logger.error(f"❌ Error ending Twilio call {call_id}: {e}")
            return False
    
    def get_call_status(self, call_id: str, account_sid: Optional[str] = None, auth_token: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Get Twilio call status
        
        Args:
            call_id: Twilio Call SID
            account_sid: Optional custom Twilio Account SID
            auth_token: Optional custom Twilio Auth Token
            **kwargs: Additional parameters
            
        Returns:
            Dict with call status information
        """
        try:
            call = twilio_service.get_call_by_sid(call_id)
            return {
                "call_id": call.sid,
                "status": call.status,
                "duration": call.duration,
                "direction": call.direction,
                "from_number": call.from_,
                "to_number": call.to,
                "start_time": call.start_time.isoformat() if call.start_time else None,
                "end_time": call.end_time.isoformat() if call.end_time else None
            }
        except Exception as e:
            logger.error(f"❌ Error getting Twilio call status {call_id}: {e}")
            return {"call_id": call_id, "status": "unknown", "error": str(e)}
    
    def get_recording_url(self, call_id: str, account_sid: Optional[str] = None, auth_token: Optional[str] = None, **kwargs) -> Optional[str]:
        """
        Get Twilio call recording URL
        
        Args:
            call_id: Twilio Call SID
            account_sid: Optional custom Twilio Account SID
            auth_token: Optional custom Twilio Auth Token
            **kwargs: Additional parameters
            
        Returns:
            Recording URL or None
        """
        try:
            recordings = twilio_service.get_call_recordings(call_id)
            if recordings:
                # Return the first recording URL
                return recordings[0].uri.replace('.json', '.mp3')
            return None
        except Exception as e:
            logger.error(f"❌ Error getting Twilio recording URL for call {call_id}: {e}")
            return None
