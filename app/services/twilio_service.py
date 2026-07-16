"""
Twilio Service Module
Handles all Twilio-related operations including client management and API calls
"""

from twilio.rest import Client
from twilio.base.exceptions import TwilioException
from app.core.config import settings
from typing import List, Dict, Optional, Any
from app.core.logger import logger

def _build_amd_kwargs(
    machine_detection: Optional[str] = None,
    machine_detection_timeout: Optional[int] = None,
    async_amd: Optional[str] = None,
    async_amd_status_callback: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the subset of AMD kwargs to pass to client.calls.create — only includes
    params that were explicitly set, so omitting them all preserves the "no AMD" default."""
    kwargs: Dict[str, Any] = {}
    if machine_detection is not None:
        kwargs["machine_detection"] = machine_detection
    if machine_detection_timeout is not None:
        kwargs["machine_detection_timeout"] = machine_detection_timeout
    if async_amd is not None:
        kwargs["async_amd"] = async_amd
    if async_amd_status_callback is not None:
        kwargs["async_amd_status_callback"] = async_amd_status_callback
    return kwargs


class TwilioService:
    """Service class for handling Twilio operations"""
    
    def __init__(self):
        self._client = None
    
    def get_client(self):
        """Get or create Twilio client using Secret Manager credentials."""
        if self._client is None:
            from app.core.secret_manager import get_twilio_credentials
            account_sid, auth_token = get_twilio_credentials()
            self._client = Client(account_sid, auth_token)
        return self._client

    def reset_client(self) -> None:
        """Force a fresh client on next call (e.g. after credential rotation)."""
        self._client = None
    
    def get_client_with_credentials(self, account_sid: str, auth_token: str):
        """Get Twilio client with custom credentials"""
        return Client(account_sid, auth_token)

    @staticmethod
    def _normalize_url(url: Optional[str]) -> Optional[str]:
        if not url:
            return url
        return str(url).strip().rstrip("/")

    def _verify_number_webhook_configuration(
        self,
        number_obj,
        expected_voice_url: Optional[str],
        expected_status_callback_url: Optional[str],
    ) -> None:
        """
        Fail-fast verification for Twilio webhook configuration.
        """
        if expected_voice_url:
            actual_voice_url = self._normalize_url(getattr(number_obj, "voice_url", None))
            expected_voice_url_normalized = self._normalize_url(expected_voice_url)
            if actual_voice_url != expected_voice_url_normalized:
                raise Exception(
                    f"Webhook verification failed: voice_url mismatch "
                    f"(expected={expected_voice_url_normalized}, actual={actual_voice_url})"
                )
            actual_voice_method = (getattr(number_obj, "voice_method", "") or "").upper()
            if actual_voice_method != "POST":
                raise Exception(
                    f"Webhook verification failed: voice_method must be POST (actual={actual_voice_method})"
                )

        if expected_status_callback_url:
            actual_status_url = self._normalize_url(getattr(number_obj, "status_callback", None))
            expected_status_url_normalized = self._normalize_url(expected_status_callback_url)
            if actual_status_url != expected_status_url_normalized:
                raise Exception(
                    f"Webhook verification failed: status_callback mismatch "
                    f"(expected={expected_status_url_normalized}, actual={actual_status_url})"
                )
            actual_status_method = (getattr(number_obj, "status_callback_method", "") or "").upper()
            if actual_status_method != "POST":
                raise Exception(
                    f"Webhook verification failed: status_callback_method must be POST "
                    f"(actual={actual_status_method})"
                )
    
    def make_call(
        self,
        to_number,
        from_number,
        webhook_url,
        status_callback_url,
        record=True,
        machine_detection: Optional[str] = None,
        machine_detection_timeout: Optional[int] = None,
        async_amd: Optional[str] = None,
        async_amd_status_callback: Optional[str] = None,
    ):
        """Make an outbound call with improved reliability and optional recording.

        AMD (Answering Machine Detection) is opt-in via machine_detection — leaving it
        None preserves the existing "no AMD" behaviour (instant TwiML, no announcements).
        """
        client = self.get_client()

        # Set up recording status callback URL (use settings for correct base URL)
        from app.core.config import settings
        recording_status_callback_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/recording-status"

        amd_kwargs = _build_amd_kwargs(
            machine_detection, machine_detection_timeout, async_amd, async_amd_status_callback
        )

        call = client.calls.create(
            to=to_number,
            from_=from_number,
            url=webhook_url,
            status_callback=status_callback_url,
            status_callback_event=['initiated', 'ringing', 'answered', 'completed'],
            status_callback_method='POST',
            record=record,  # Enable call recording
            recording_channels='dual',  # Record both channels
            recording_status_callback=recording_status_callback_url,  # Get recording status updates
            timeout=30,  # Answer timeout (30 seconds)
            **amd_kwargs,
        )

        return call

    def make_call_with_credentials(
        self,
        to_number: str,
        from_number: str,
        webhook_url: str,
        status_callback_url: str,
        account_sid: str,
        auth_token: str,
        record: bool = True,
        machine_detection: Optional[str] = None,
        machine_detection_timeout: Optional[int] = None,
        async_amd: Optional[str] = None,
        async_amd_status_callback: Optional[str] = None,
    ):
        """Make call with custom Twilio credentials"""
        client = self.get_client_with_credentials(account_sid, auth_token)

        from app.core.config import settings
        recording_status_callback_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/recording-status"

        amd_kwargs = _build_amd_kwargs(
            machine_detection, machine_detection_timeout, async_amd, async_amd_status_callback
        )

        call = client.calls.create(
            to=to_number,
            from_=from_number,
            url=webhook_url,
            status_callback=status_callback_url,
            status_callback_event=['initiated', 'ringing', 'answered', 'completed'],
            status_callback_method='POST',
            record=record,
            recording_channels='dual',
            recording_status_callback=recording_status_callback_url,
            timeout=30,
            **amd_kwargs,
        )

        return call
    
    def get_recent_calls(self, limit=10):
        """Get recent calls"""
        client = self.get_client()
        return client.calls.list(limit=limit)
    
    def get_call_by_sid(self, call_sid):
        """Get a specific call by SID"""
        client = self.get_client()
        return client.calls(call_sid).fetch()
    
    def get_call_recordings(self, call_sid):
        """Get recordings for a specific call"""
        client = self.get_client()
        return client.calls(call_sid).recordings.list()
    
    def has_call_recordings(self, call_sid):
        """Check if a call has recordings"""
        try:
            recordings = self.get_call_recordings(call_sid)
            return len(recordings) > 0
        except Exception:
            return False
    
    def get_recording_url(self, recording_sid):
        """Get the URL for a specific recording"""
        client = self.get_client()
        recording = client.recordings(recording_sid).fetch()
        
        # Check if recording is ready
        if recording.status != 'completed':
            return {
                'sid': recording.sid,
                'url': None,
                'duration': recording.duration,
                'channels': recording.channels,
                'status': recording.status,
                'date_created': recording.date_created,
                'date_updated': recording.date_updated,
                'message': f"Recording is {recording.status}. URL will be available when processing is complete."
            }
        
        # Recording is completed, return the URL
        return {
            'sid': recording.sid,
            'url': f"https://api.twilio.com{recording.uri.replace('.json', '.mp3')}",
            'duration': recording.duration,
            'channels': recording.channels,
            'status': recording.status,
            'date_created': recording.date_created,
            'date_updated': recording.date_updated
        }
    
    def get_phone_number(self):
        """Get the configured Twilio phone number"""
        phone_number = settings.TWILIO_PHONE_NUMBER
        if not phone_number or phone_number == "+1234567890":
            raise Exception("Please configure a valid TWILIO_PHONE_NUMBER in your settings")
        return phone_number
    
    def validate_phone_number(self, phone_number):
        """Validate phone number format"""
        if not phone_number or not phone_number.startswith('+'):
            return False
        return True

    # Phone Number Purchasing Methods
    
    def search_available_numbers(
        self,
        country_code: str = "US",
        number_type: str = "local",
        area_code: Optional[str] = None,
        contains: Optional[str] = None,
        voice_enabled: bool = True,
        sms_enabled: bool = True,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Search for available phone numbers
        
        Args:
            country_code: Country code (e.g., "US", "CA", "GB")
            area_code: Specific area code to search for
            contains: Pattern to search for in phone numbers
            voice_enabled: Whether numbers should support voice
            sms_enabled: Whether numbers should support SMS
            limit: Maximum number of results to return
            
        Returns:
            List of available phone numbers with details
        """
        client = self.get_client()
        
        try:
            # Build search parameters
            search_params = {
                'voice_enabled': voice_enabled,
                'sms_enabled': sms_enabled,
                'limit': limit
            }
            
            if area_code:
                search_params['area_code'] = area_code
            if contains:
                search_params['contains'] = contains
            
            # Search for available numbers by type
            phone_numbers_resource = client.available_phone_numbers(country_code)
            if number_type == "toll_free":
                available_numbers = phone_numbers_resource.toll_free.list(**search_params)
            elif number_type == "mobile":
                available_numbers = phone_numbers_resource.mobile.list(**search_params)
            else:
                available_numbers = phone_numbers_resource.local.list(**search_params)
            
            # Format results
            results = []
            for number in available_numbers:
                results.append({
                    'phone_number': number.phone_number,
                    'friendly_name': getattr(number, 'friendly_name', None),
                    'locality': getattr(number, 'locality', None),
                    'region': getattr(number, 'region', None),
                    'country': country_code,  # Use the input country_code
                    'capabilities': {
                        'voice': number.capabilities.get('voice', False),
                        'sms': number.capabilities.get('sms', False),
                        'mms': number.capabilities.get('mms', False)
                    },
                    'beta': getattr(number, 'beta', False)
                })
            
            return results
            
        except TwilioException as e:
            raise Exception(f"Error searching for available numbers: {str(e)}")
    
    def purchase_phone_number(self, phone_number: str, webhook_url: Optional[str] = None,
                             status_callback_url: Optional[str] = None,
                             status_callback_method: str = "POST") -> Dict[str, Any]:
        """
        Purchase a phone number
        
        Args:
            phone_number: The phone number to purchase (e.g., "+1234567890")
            webhook_url: Webhook URL for incoming calls
            status_callback_url: Webhook URL for call status updates
            status_callback_method: HTTP method for status callbacks
            
        Returns:
            Dictionary with purchase details
        """
        client = self.get_client()
        
        try:
            # Build purchase parameters
            purchase_params = {
                'phone_number': phone_number
            }
            
            if webhook_url:
                purchase_params['voice_url'] = webhook_url
                purchase_params['voice_method'] = 'POST'
            
            if status_callback_url:
                purchase_params['status_callback'] = status_callback_url
                purchase_params['status_callback_method'] = status_callback_method
            
            # Purchase the number
            incoming_phone_number = client.incoming_phone_numbers.create(**purchase_params)
            # Verify webhooks (fail-fast) when requested by caller
            self._verify_number_webhook_configuration(
                incoming_phone_number, webhook_url, status_callback_url
            )
            
            return {
                'sid': incoming_phone_number.sid,
                'phone_number': incoming_phone_number.phone_number,
                'friendly_name': incoming_phone_number.friendly_name,
                'voice_url': incoming_phone_number.voice_url,
                'voice_method': incoming_phone_number.voice_method,
                'status_callback': incoming_phone_number.status_callback,
                'status_callback_method': incoming_phone_number.status_callback_method,
                'capabilities': incoming_phone_number.capabilities,
                'date_created': str(incoming_phone_number.date_created),
                'date_updated': str(incoming_phone_number.date_updated)
            }
            
        except TwilioException as e:
            raise Exception(f"Error purchasing phone number: {str(e)}")
    
    def list_owned_numbers(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List all phone numbers owned by the account
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of owned phone numbers
        """
        client = self.get_client()
        
        try:
            incoming_phone_numbers = client.incoming_phone_numbers.list(limit=limit)
            
            results = []
            for number in incoming_phone_numbers:
                results.append({
                    # 'sid': number.sid,
                    'phone_number': number.phone_number,
                    'friendly_name': number.friendly_name,
                    # 'voice_url': number.voice_url,
                    # 'voice_method': number.voice_method,
                    # 'status_callback': number.status_callback,
                    # 'status_callback_method': number.status_callback_method,
                    # 'capabilities': number.capabilities,
                    # 'date_created': str(number.date_created),
                    # 'date_updated': str(number.date_updated)
                })
            
            return results
            
        except TwilioException as e:
            raise Exception(f"Error listing owned numbers: {str(e)}")
    
    def get_number_details(self, phone_number_sid: str) -> Dict[str, Any]:
        """
        Get details of a specific phone number
        
        Args:
            phone_number_sid: The SID of the phone number
            
        Returns:
            Dictionary with phone number details
        """
        client = self.get_client()
        
        try:
            number = client.incoming_phone_numbers(phone_number_sid).fetch()
            
            return {
                'sid': number.sid,
                'phone_number': number.phone_number,
                'friendly_name': number.friendly_name,
                'voice_url': number.voice_url,
                'voice_method': number.voice_method,
                'status_callback': number.status_callback,
                'status_callback_method': number.status_callback_method,
                'capabilities': number.capabilities,
                'date_created': str(number.date_created),
                'date_updated': str(number.date_updated)
            }
            
        except TwilioException as e:
            raise Exception(f"Error fetching number details: {str(e)}")
    
    def update_number_configuration(self, phone_number_sid: str, 
                                  friendly_name: Optional[str] = None,
                                  webhook_url: Optional[str] = None,
                                  status_callback_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Update configuration for a phone number
        
        Args:
            phone_number_sid: The SID of the phone number
            friendly_name: New friendly name for the number
            webhook_url: New webhook URL for incoming calls
            status_callback_url: New webhook URL for status updates
            
        Returns:
            Dictionary with updated phone number details
        """
        client = self.get_client()
        
        try:
            update_params = {}
            
            if friendly_name is not None:
                update_params['friendly_name'] = friendly_name
            if webhook_url is not None:
                update_params['voice_url'] = webhook_url
                update_params['voice_method'] = 'POST'
            if status_callback_url is not None:
                update_params['status_callback'] = status_callback_url
                update_params['status_callback_method'] = 'POST'
            
            if not update_params:
                raise Exception("No parameters provided for update")
            
            number = client.incoming_phone_numbers(phone_number_sid).update(**update_params)
            self._verify_number_webhook_configuration(
                number, webhook_url, status_callback_url
            )
            
            return {
                'sid': number.sid,
                'phone_number': number.phone_number,
                'friendly_name': number.friendly_name,
                'voice_url': number.voice_url,
                'voice_method': number.voice_method,
                'status_callback': number.status_callback,
                'status_callback_method': number.status_callback_method,
                'capabilities': number.capabilities,
                'date_created': str(number.date_created),
                'date_updated': str(number.date_updated)
            }
            
        except TwilioException as e:
            raise Exception(f"Error updating number configuration: {str(e)}")

    def update_number_configuration_with_credentials(
        self,
        phone_number_sid: str,
        account_sid: str,
        auth_token: str,
        friendly_name: Optional[str] = None,
        webhook_url: Optional[str] = None,
        status_callback_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update configuration for a phone number using custom Twilio credentials.
        """
        client = self.get_client_with_credentials(account_sid, auth_token)

        try:
            update_params = {}

            if friendly_name is not None:
                update_params["friendly_name"] = friendly_name
            if webhook_url is not None:
                update_params["voice_url"] = webhook_url
                update_params["voice_method"] = "POST"
            if status_callback_url is not None:
                update_params["status_callback"] = status_callback_url
                update_params["status_callback_method"] = "POST"

            if not update_params:
                raise Exception("No parameters provided for update")

            number = client.incoming_phone_numbers(phone_number_sid).update(**update_params)
            self._verify_number_webhook_configuration(
                number, webhook_url, status_callback_url
            )

            return {
                "sid": number.sid,
                "phone_number": number.phone_number,
                "friendly_name": number.friendly_name,
                "voice_url": number.voice_url,
                "voice_method": number.voice_method,
                "status_callback": number.status_callback,
                "status_callback_method": number.status_callback_method,
                "capabilities": number.capabilities,
                "date_created": str(number.date_created),
                "date_updated": str(number.date_updated),
            }

        except TwilioException as e:
            raise Exception(f"Error updating number configuration: {str(e)}")
    
    def release_phone_number(self, phone_number_sid: str) -> bool:
        """
        Release (delete) a phone number
        
        Args:
            phone_number_sid: The SID of the phone number to release
            
        Returns:
            True if successful
        """
        client = self.get_client()
        
        try:
            client.incoming_phone_numbers(phone_number_sid).delete()
            return True
            
        except TwilioException as e:
            raise Exception(f"Error releasing phone number: {str(e)}")
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information including balance and limits
        
        Returns:
            Dictionary with account details
        """
        client = self.get_client()
        
        try:
            account = client.api.accounts(settings.TWILIO_ACCOUNT_SID).fetch()
            
            return {
                'sid': account.sid,
                'friendly_name': account.friendly_name,
                'status': account.status,
                'type': account.type,
                'date_created': str(account.date_created),
                'date_updated': str(account.date_updated)
            }
            
        except TwilioException as e:
            raise Exception(f"Error fetching account info: {str(e)}")
    
    def end_call(self, call_sid: str) -> bool:
        """
        End a call programmatically using Twilio API
        
        Args:
            call_sid: The SID of the call to end
            
        Returns:
            True if successful, False otherwise
        """
        client = self.get_client()
        
        try:
            call = client.calls(call_sid).update(status='completed')
            logger.info(f"✅ Call {call_sid} ended successfully")
            return True
            
        except TwilioException as e:
            logger.error(f"❌ Error ending call {call_sid}: {str(e)}")
            return False

    def end_call_with_credentials(self, call_sid: str, account_sid: str, auth_token: str) -> bool:
        """
        End a call using explicit Twilio credentials.
        """
        client = self.get_client_with_credentials(account_sid, auth_token)

        try:
            client.calls(call_sid).update(status='completed')
            logger.info(f"✅ Call {call_sid} ended successfully with explicit credentials")
            return True

        except TwilioException as e:
            logger.error(f"❌ Error ending call {call_sid} with explicit credentials: {str(e)}")
            return False

    def start_recording_with_credentials(self, call_sid: str, account_sid: str, auth_token: str) -> bool:
        """
        Start call recording using explicit Twilio credentials.
        Recording status updates are sent to the existing recording-status webhook.
        """
        client = self.get_client_with_credentials(account_sid, auth_token)
        recording_status_callback_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/recording-status"

        try:
            client.calls(call_sid).recordings.create(
                recording_channels="dual",
                recording_status_callback=recording_status_callback_url,
                recording_status_callback_method="POST",
            )
            logger.info(f"✅ Recording started for call {call_sid} with explicit credentials")
            return True

        except TwilioException as e:
            logger.error(f"❌ Error starting recording for call {call_sid}: {str(e)}")
            return False
    
    def redirect_call(self, call_sid: str, redirect_url: str, method: str = "POST") -> bool:
        """
        Redirect an in-progress call to a new TwiML URL
        This is useful for interrupting media streams to play responses
        
        Args:
            call_sid: The SID of the call to redirect
            redirect_url: The URL to fetch new TwiML from
            method: HTTP method to use (POST or GET)
            
        Returns:
            True if successful, False otherwise
        """
        client = self.get_client()
        
        try:
            call = client.calls(call_sid).update(
                url=redirect_url,
                method=method
            )
            logger.info(f"✅ Call {call_sid} redirected to {redirect_url}")
            return True
            
        except TwilioException as e:
            logger.error(f"❌ Error redirecting call {call_sid}: {str(e)}")
            return False

    def update_call_twiml(
        self,
        call_sid: str,
        twiml: str,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
    ) -> bool:
        """
        Inject TwiML directly into an in-progress call (no URL fetch round-trip).
        Used by the AMD callback to play a voicemail message and hang up.
        """
        client = (
            self.get_client_with_credentials(account_sid, auth_token)
            if account_sid and auth_token
            else self.get_client()
        )

        try:
            client.calls(call_sid).update(twiml=twiml)
            logger.info(f"✅ Call {call_sid} updated with inline TwiML")
            return True
        except TwilioException as e:
            logger.error(f"❌ Error updating call {call_sid} with TwiML: {str(e)}")
            return False

    def redirect_call_with_credentials(
        self,
        call_sid: str,
        redirect_url: str,
        account_sid: str,
        auth_token: str,
        method: str = "POST",
    ) -> bool:
        """Redirect an in-progress call using explicit Twilio credentials (multi-tenant)."""
        client = self.get_client_with_credentials(account_sid, auth_token)
        try:
            client.calls(call_sid).update(url=redirect_url, method=method)
            logger.info("Call %s redirected (custom creds) to %s", call_sid, redirect_url)
            return True
        except TwilioException as e:
            logger.error("Error redirecting call %s with custom creds: %s", call_sid, e)
            return False

# Global instance
twilio_service = TwilioService()