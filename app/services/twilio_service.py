"""
Twilio Service Module
Handles all Twilio-related operations including client management and API calls
"""

from twilio.rest import Client
from twilio.base.exceptions import TwilioException
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any

# Load environment variables
load_dotenv()

class TwilioService:
    """Service class for handling Twilio operations"""
    
    def __init__(self):
        self._client = None
    
    def get_client(self):
        """Get or create Twilio client"""
        if self._client is None:
            account_sid = os.getenv("TWILIO_ACCOUNT_SID")
            auth_token = os.getenv("TWILIO_AUTH_TOKEN")
            
            if not account_sid or not auth_token:
                raise Exception("Twilio credentials not found. Please set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in your .env file.")
            
            self._client = Client(account_sid, auth_token)
        
        return self._client
    
    def make_call(self, to_number, from_number, webhook_url, status_callback_url):
        """Make an outbound call"""
        client = self.get_client()
        
        call = client.calls.create(
            to=to_number,
            from_=from_number,
            url=webhook_url,
            status_callback=status_callback_url,
            status_callback_event=['initiated', 'ringing', 'answered', 'completed'],
            status_callback_method='POST'
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
    
    def get_phone_number(self):
        """Get the configured Twilio phone number"""
        return os.getenv("TWILIO_PHONE_NUMBER")
    
    def validate_phone_number(self, phone_number):
        """Validate phone number format"""
        if not phone_number or not phone_number.startswith('+'):
            return False
        return True

    # Phone Number Purchasing Methods
    
    def search_available_numbers(self, country_code: str = "US", area_code: Optional[str] = None, 
                                contains: Optional[str] = None, voice_enabled: bool = True,
                                sms_enabled: bool = True, limit: int = 20) -> List[Dict[str, Any]]:
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
            
            # Search for available numbers
            available_numbers = client.available_phone_numbers(country_code).local.list(**search_params)
            
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
            account = client.api.accounts(os.getenv("TWILIO_ACCOUNT_SID")).fetch()
            
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

# Global instance
twilio_service = TwilioService()