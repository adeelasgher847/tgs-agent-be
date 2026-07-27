"""
Base CRM Service Interface
All CRM services (Monday.com, ClickUp, Jira, Trello) must implement this interface
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseCRMService(ABC):
    """Base interface for all CRM services"""
    
    @abstractmethod
    def get_api_key(self) -> str:
        """Get decrypted API key for this CRM"""
        pass
    
    @abstractmethod
    def build_container_url(self, container_id: str) -> str:
        """Build URL for the container (board/list/project)"""
        pass
    
    @abstractmethod
    def create_container(self, container_name: str, **kwargs) -> Dict[str, str]:
        """
        Create a container (board/list/project) in the CRM.
        
        Returns:
            Dict with 'id' and 'url' keys
        """
        pass
    
    @abstractmethod
    def ensure_required_fields(self, container_id: str) -> Dict[str, str]:
        """
        Ensure the container has all required fields/columns.
        Creates missing fields if needed.
        
        Returns:
            Dict mapping field keys to their CRM field IDs
            Example: {"status": "status_col_123", "agent_id": "agent_col_456", ...}
        """
        pass
    
    @abstractmethod
    def create_scheduled_call_item(
        self,
        container_id: str,
        field_map: Dict[str, str],
        phone_number: str,
        agent_id: str,
        call_time_utc: str,
        tenant_id: str,
        user_id: str,
        batch_id: Optional[str] = None,
        phone_number_id: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Create a scheduled call item/task/issue/card in the CRM.
        
        Returns:
            Dict with item info (at minimum: {"id": "item_id"}) or None if failed
        """
        pass
    
    @abstractmethod
    def update_item_status(
        self,
        container_id: str,
        item_id: str,
        status: str,
        field_map: Dict[str, str],
    ) -> Optional[dict]:
        """
        Update the status of an item.
        
        Args:
            container_id: Container ID (board/list/project)
            item_id: Item ID (item/task/issue/card)
            status: Status to set ("Pending", "Called", "Failed")
            field_map: Field mapping dictionary
            
        Returns:
            Updated item data or None if failed
        """
        pass
    
    @abstractmethod
    def update_item_call_session_id(
        self,
        container_id: str,
        item_id: str,
        call_session_id: str,
        field_map: Dict[str, str],
    ) -> Optional[dict]:
        """Update call_session_id field for an item"""
        pass
    
    @abstractmethod
    def get_required_fields(self) -> List[Dict]:
        """
        Get list of required fields/columns for scheduled calls.
        
        Returns:
            List of field definitions with keys: key, title, type, defaults (optional)
        """
        pass
    
    @abstractmethod
    def delete_items_by_tenant(
        self,
        container_id: str,
        tenant_id: str,
        field_map: Dict[str, str],
        batch_size: int = 50
    ) -> int:
        """
        Delete items from container that belong to a specific tenant.
        Filters by tenant_id field/column.
        
        Args:
            container_id: Container ID (board/list/project)
            tenant_id: Tenant ID to filter by (UUID string)
            field_map: Field mapping dictionary (must include "tenant_id")
            batch_size: Number of items to fetch per batch
            
        Returns:
            Number of items deleted
        """
        pass
    
    @abstractmethod
    def count_pending_items_for_tenant(
        self,
        container_id: str,
        tenant_id: str,
        field_map: Dict[str, str],
        pending_label: str = "Pending",
        batch_size: int = 100
    ) -> int:
        """
        Count items for a given tenant that are still in 'Pending' status.
        
        Args:
            container_id: Container ID (board/list/project)
            tenant_id: Tenant ID to filter by (UUID string)
            field_map: Field mapping dictionary (must include "tenant_id" and "status")
            pending_label: The label/text used for pending status (default: "Pending")
            batch_size: Number of items to fetch per batch
            
        Returns:
            Number of items with status == pending_label for the given tenant
        """
        pass

