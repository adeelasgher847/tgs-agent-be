"""
Base Dialer Service
Abstract base class for all dialer services (Twilio, Vicidial, etc.)
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from app.core.logger import logger

class BaseDialerService(ABC):
    """Abstract base class for dialer services"""
    
    @abstractmethod
    def initiate_call(
        self,
        to_number: str,
        from_number: str,
        webhook_url: str,
        status_callback_url: str,
        call_session_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Initiate a call using the dialer
        
        Args:
            to_number: Destination phone number
            from_number: Source phone number
            webhook_url: Webhook URL for call events
            status_callback_url: Status callback URL
            call_session_id: Call session ID
            **kwargs: Additional dialer-specific parameters
            
        Returns:
            Dict with call information (call_id, status, etc.)
        """
        pass
    
    @abstractmethod
    def end_call(self, call_id: str, **kwargs) -> bool:
        """
        End an active call
        
        Args:
            call_id: External call ID (Twilio SID, Vicidial call ID, etc.)
            **kwargs: Additional dialer-specific parameters
            
        Returns:
            bool: True if call ended successfully
        """
        pass
    
    @abstractmethod
    def get_call_status(self, call_id: str, **kwargs) -> Dict[str, Any]:
        """
        Get call status
        
        Args:
            call_id: External call ID
            **kwargs: Additional dialer-specific parameters
            
        Returns:
            Dict with call status information
        """
        pass
    
    @abstractmethod
    def get_recording_url(self, call_id: str, **kwargs) -> Optional[str]:
        """
        Get call recording URL
        
        Args:
            call_id: External call ID
            **kwargs: Additional dialer-specific parameters
            
        Returns:
            Recording URL or None
        """
        pass
