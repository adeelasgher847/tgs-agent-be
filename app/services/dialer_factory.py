"""
Dialer Factory
Factory to get appropriate dialer service based on dialer_type
"""

from typing import Optional
from app.services.base_dialer_service import BaseDialerService
from app.services.twilio_dialer_service import TwilioDialerService
from app.services.vicidial_dialer_service import VicidialDialerService
from app.core.logger import logger


class DialerFactory:
    """Factory for creating dialer service instances"""
    
    _twilio_dialer: Optional[TwilioDialerService] = None
    _vicidial_dialer: Optional[VicidialDialerService] = None
    
    @classmethod
    def get_dialer(cls, dialer_type: str) -> BaseDialerService:
        """
        Get dialer service instance based on dialer_type
        
        Args:
            dialer_type: "twilio" or "vicidial"
            
        Returns:
            BaseDialerService instance
            
        Raises:
            ValueError: If dialer_type is not supported
        """
        dialer_type = dialer_type.lower()
        
        if dialer_type == "twilio":
            if cls._twilio_dialer is None:
                cls._twilio_dialer = TwilioDialerService()
            return cls._twilio_dialer
        
        elif dialer_type == "vicidial":
            if cls._vicidial_dialer is None:
                cls._vicidial_dialer = VicidialDialerService()
            return cls._vicidial_dialer
        
        else:
            logger.error(f"❌ Unsupported dialer_type: {dialer_type}")
            raise ValueError(f"Unsupported dialer_type: {dialer_type}. Supported types: 'twilio', 'vicidial'")
