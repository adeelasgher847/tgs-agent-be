"""
CRM Service Factory - Creates appropriate CRM service based on type
"""

from typing import Optional
from app.services.base_crm_service import BaseCRMService
from app.services.monday_service import MondayService
from app.services.clickup_service import ClickUpService
from app.services.jira_service import JiraService
from app.services.trello_service import TrelloService
from app.models.tenant_crm_config import CRMConfig
from app.core.security import decrypt_api_key
import json


class CRMServiceFactory:
    """Factory for creating CRM service instances"""
    
    @staticmethod
    def get_service(crm_config: CRMConfig) -> BaseCRMService:
        """
        Get CRM service instance based on CRM config.
        
        Args:
            crm_config: CRMConfig model instance
            
        Returns:
            BaseCRMService instance
        """
        crm_type = crm_config.crm_type.lower()
        api_key = decrypt_api_key(crm_config.encrypted_api_key)
        
        if crm_type == "monday":
            # MondayService accepts optional API key
            service = MondayService(api_key=crm_config.encrypted_api_key)  # Pass encrypted, service will decrypt
            return service
        elif crm_type == "clickup":
            return ClickUpService(api_key=crm_config.encrypted_api_key)  # Pass encrypted, service will decrypt
        elif crm_type == "jira":
            # Jira needs email and server_url from additional_config
            additional_config = json.loads(crm_config.additional_config) if crm_config.additional_config else {}
            email = additional_config.get("email", "")
            server_url = additional_config.get("server_url", "")
            return JiraService(
                api_key=crm_config.encrypted_api_key,
                email=email,
                server_url=server_url
            )
        elif crm_type == "trello":
            # Trello needs both API key and token
            additional_config = json.loads(crm_config.additional_config) if crm_config.additional_config else {}
            api_token = additional_config.get("api_token", "")
            # API token is encrypted, TrelloService will decrypt it automatically
            return TrelloService(
                api_key=crm_config.encrypted_api_key,
                api_token=api_token  # Encrypted token, service will decrypt
            )
        else:
            raise ValueError(f"Unknown CRM type: {crm_type}")
    
    @staticmethod
    def get_service_by_type(crm_type: str, api_key: str, **kwargs) -> BaseCRMService:
        """
        Get CRM service instance by type (for testing or direct usage).
        
        Args:
            crm_type: "monday" | "clickup" | "jira" | "trello"
            api_key: Encrypted or plain API key
            **kwargs: Additional parameters (email, server_url for Jira, api_token for Trello)
        """
        crm_type = crm_type.lower()
        
        if crm_type == "monday":
            return MondayService(api_key=api_key)
        elif crm_type == "clickup":
            return ClickUpService(api_key=api_key)
        elif crm_type == "jira":
            return JiraService(
                api_key=api_key,
                email=kwargs.get("email", ""),
                server_url=kwargs.get("server_url", "")
            )
        elif crm_type == "trello":
            return TrelloService(
                api_key=api_key,
                api_token=kwargs.get("api_token", "")
            )
        else:
            raise ValueError(f"Unknown CRM type: {crm_type}")

