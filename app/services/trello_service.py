"""
Trello API Service for Scheduled Calls Integration
"""

import json
from typing import Dict, List, Optional
import requests
from app.services.base_crm_service import BaseCRMService
from app.core.security import decrypt_api_key


class TrelloService(BaseCRMService):
    """Service for interacting with Trello API"""

    API_URL = "https://api.trello.com/1"
    REQUIRED_FIELDS = [
        {"key": "status", "title": "Status", "type": "dropdown"},
        {"key": "agent_id", "title": "Agent ID", "type": "text"},
        {"key": "call_time_utc", "title": "Call Time UTC", "type": "text"},
        {"key": "tenant_id", "title": "Tenant ID", "type": "text"},
        {"key": "user_id", "title": "User ID", "type": "text"},
        {"key": "batch_id", "title": "Batch ID", "type": "text"},
        {"key": "call_session_id", "title": "Call Session ID", "type": "text"},
        {"key": "phone_number_id", "title": "Phone Number ID", "type": "text"},
        {"key": "email_sent", "title": "Email Sent", "type": "dropdown"},
    ]

    def __init__(self, api_key: str, api_token: str):
        """
        Initialize Trello service
        
        Args:
            api_key: Trello API key
            api_token: Trello API token
        """
        self.api_key = api_key
        self.api_token = api_token

    def get_api_key(self) -> str:
        """Get decrypted API key"""
        return decrypt_api_key(self.api_key) if self.api_key.startswith("eyJ") else self.api_key

    def get_api_token(self) -> str:
        """Get decrypted API token"""
        return decrypt_api_key(self.api_token) if self.api_token.startswith("eyJ") else self.api_token

    def build_container_url(self, container_id: str) -> str:
        """Build URL for Trello board"""
        return f"https://trello.com/b/{container_id}"

    def _auth_params(self) -> Dict[str, str]:
        """Get authentication parameters"""
        return {
            "key": self.get_api_key(),
            "token": self.get_api_token(),
        }

    def create_container(self, container_name: str, **kwargs) -> Dict[str, str]:
        """
        Create a Trello board for scheduled calls.
        """
        url = f"{self.API_URL}/boards"
        params = self._auth_params()
        params.update({
            "name": container_name,
            "defaultLists": "false",
        })
        
        response = requests.post(url, params=params, timeout=20)
        response.raise_for_status()
        board_data = response.json()
        
        board_id = board_data.get("shortLink", "")  # Use shortLink as ID
        return {
            "id": board_id,
            "url": self.build_container_url(board_id),
        }

    def ensure_required_fields(self, container_id: str) -> Dict[str, str]:
        """
        Ensure Trello board has required custom fields (Power-Ups).
        Note: Custom fields in Trello require Power-Ups. This method verifies they exist.
        """
        # Get custom fields for board
        url = f"{self.API_URL}/boards/{container_id}/customFields"
        params = self._auth_params()
        
        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            existing_fields = response.json()
        except:
            existing_fields = []
        
        # Map fields by name
        field_map = {}
        field_by_name = {f.get("name", "").lower(): f for f in existing_fields}
        
        # Check for required fields
        for field_def in self.REQUIRED_FIELDS:
            field_name = field_def["title"]
            field_key = field_def["key"]
            
            if field_name.lower() in field_by_name:
                field_map[field_key] = field_by_name[field_name.lower()].get("id", "")
            else:
                # Try to create custom field (requires Power-Ups)
                # Note: This may fail if Power-Ups are not enabled
                try:
                    create_url = f"{self.API_URL}/customFields"
                    create_params = self._auth_params()
                    create_params.update({
                        "idModel": container_id,
                        "modelType": "board",
                        "name": field_name,
                        "type": "text" if field_def["type"] == "text" else "list",
                    })
                    
                    create_response = requests.post(create_url, params=create_params, timeout=20)
                    if create_response.status_code == 200:
                        created_field = create_response.json()
                        field_map[field_key] = created_field.get("id", "")
                except Exception as e:
                    print(f"⚠️ Failed to create Trello custom field {field_name}: {e}")
        
        return field_map

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
        """Create a scheduled call card in Trello board"""
        # Get default list (or create one)
        url = f"{self.API_URL}/boards/{container_id}/lists"
        params = self._auth_params()
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        lists = response.json()
        
        if not lists:
            # Create a default list
            create_list_url = f"{self.API_URL}/lists"
            create_params = self._auth_params()
            create_params.update({
                "name": "Scheduled Calls",
                "idBoard": container_id,
            })
            create_response = requests.post(create_list_url, params=create_params, timeout=20)
            if create_response.status_code == 200:
                lists = [create_response.json()]
        
        list_id = lists[0].get("id", "") if lists else ""
        
        # Create card
        create_card_url = f"{self.API_URL}/cards"
        card_params = self._auth_params()
        card_params.update({
            "name": phone_number,
            "desc": f"Scheduled call at {call_time_utc}",
            "idList": list_id,
        })
        
        try:
            response = requests.post(create_card_url, params=card_params, timeout=20)
            response.raise_for_status()
            card_data = response.json()
            card_id = card_data.get("id", "")
            
            # Add custom field values
            for key, field_id in field_map.items():
                if key == "status":
                    # Update card label or custom field
                    update_url = f"{self.API_URL}/cards/{card_id}/customField/{field_id}/item"
                    update_params = self._auth_params()
                    update_params.update({
                        "value": {"text": "Pending"}
                    })
                    requests.put(update_url, params=update_params, timeout=20)
                elif key in ["agent_id", "call_time_utc", "tenant_id", "user_id"]:
                    # Add as card description or custom field
                    value = {
                        "agent_id": agent_id,
                        "call_time_utc": call_time_utc,
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                    }.get(key, "")
                    
                    if field_id:
                        update_url = f"{self.API_URL}/cards/{card_id}/customField/{field_id}/item"
                        update_params = self._auth_params()
                        update_params.update({
                            "value": {"text": value}
                        })
                        requests.put(update_url, params=update_params, timeout=20)
            
            return card_data
        except Exception as exc:
            print(f"⚠️ Failed to create Trello card for {phone_number}: {exc}")
            return None

    def update_item_status(
        self,
        container_id: str,
        item_id: str,
        status: str,
        field_map: Dict[str, str],
    ) -> Optional[dict]:
        """Update card status in Trello"""
        status_field_id = field_map.get("status")
        if not status_field_id:
            return None
        
        url = f"{self.API_URL}/cards/{item_id}/customField/{status_field_id}/item"
        params = self._auth_params()
        params.update({
            "value": {"text": status}
        })
        
        try:
            response = requests.put(url, params=params, timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            print(f"⚠️ Failed to update Trello card {item_id} status: {exc}")
            return None

    def update_item_call_session_id(
        self,
        container_id: str,
        item_id: str,
        call_session_id: str,
        field_map: Dict[str, str],
    ) -> Optional[dict]:
        """Update call_session_id field for a Trello card"""
        session_field_id = field_map.get("call_session_id")
        if not session_field_id:
            return None
        
        url = f"{self.API_URL}/cards/{item_id}/customField/{session_field_id}/item"
        params = self._auth_params()
        params.update({
            "value": {"text": call_session_id}
        })
        
        try:
            response = requests.put(url, params=params, timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            print(f"⚠️ Failed to update call_session_id for Trello card {item_id}: {exc}")
            return None

    def get_required_fields(self) -> List[Dict]:
        """Get list of required fields"""
        return self.REQUIRED_FIELDS

    def delete_items_by_tenant(
        self,
        container_id: str,
        tenant_id: str,
        field_map: Dict[str, str],
        batch_size: int = 50
    ) -> int:
        """Delete cards from Trello board that belong to a specific tenant"""
        tenant_field_id = field_map.get("tenant_id")
        if not tenant_field_id:
            raise ValueError("tenant_id field not found in field map")
        
        deleted = 0
        
        # Get all cards from board
        url = f"{self.API_URL}/boards/{container_id}/cards"
        params = self._auth_params()
        params.update({"filter": "all"})
        
        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            cards = response.json()
        except Exception as exc:
            print(f"⚠️ Failed to fetch Trello cards: {exc}")
            return 0
        
        for card in cards:
            card_id = card.get("id", "")
            
            # Get custom fields for this card
            custom_fields_url = f"{self.API_URL}/card/{card_id}/customFields"
            custom_fields_params = self._auth_params()
            
            try:
                custom_fields_response = requests.get(custom_fields_url, params=custom_fields_params, timeout=20)
                custom_fields_response.raise_for_status()
                custom_fields = custom_fields_response.json()
            except Exception as exc:
                print(f"⚠️ Failed to get custom fields for card {card_id}: {exc}")
                continue
            
            # Check tenant_id field
            item_tenant_id = None
            for field in custom_fields:
                if field.get("id") == tenant_field_id:
                    field_value = field.get("value", {})
                    if isinstance(field_value, dict):
                        item_tenant_id = field_value.get("text", "").strip()
                    else:
                        item_tenant_id = str(field_value).strip()
                    break
            
            # Delete if tenant_id matches
            if item_tenant_id == tenant_id:
                try:
                    delete_url = f"{self.API_URL}/cards/{card_id}"
                    delete_params = self._auth_params()
                    delete_response = requests.delete(delete_url, params=delete_params, timeout=20)
                    delete_response.raise_for_status()
                    deleted += 1
                    print(f"✅ Deleted Trello card {card_id} (tenant: {tenant_id})")
                except Exception as exc:
                    print(f"⚠️ Failed to delete Trello card {card_id}: {exc}")
        
        return deleted

    def count_pending_items_for_tenant(
        self,
        container_id: str,
        tenant_id: str,
        field_map: Dict[str, str],
        pending_label: str = "Pending",
        batch_size: int = 100
    ) -> int:
        """Count pending cards from Trello board that belong to a specific tenant"""
        tenant_field_id = field_map.get("tenant_id")
        status_field_id = field_map.get("status")
        if not tenant_field_id or not status_field_id:
            raise ValueError("tenant_id or status field not found in field map")
        
        pending_count = 0
        
        # Get all cards from board
        url = f"{self.API_URL}/boards/{container_id}/cards"
        params = self._auth_params()
        params.update({"filter": "all"})
        
        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            cards = response.json()
        except Exception as exc:
            print(f"⚠️ Failed to fetch Trello cards: {exc}")
            return 0
        
        for card in cards:
            card_id = card.get("id", "")
            
            # Get custom fields for this card
            custom_fields_url = f"{self.API_URL}/card/{card_id}/customFields"
            custom_fields_params = self._auth_params()
            
            try:
                custom_fields_response = requests.get(custom_fields_url, params=custom_fields_params, timeout=20)
                custom_fields_response.raise_for_status()
                custom_fields = custom_fields_response.json()
            except Exception as exc:
                print(f"⚠️ Failed to get custom fields for card {card_id}: {exc}")
                continue
            
            # Check tenant_id and status fields
            item_tenant_id = None
            item_status = None
            for field in custom_fields:
                if field.get("id") == tenant_field_id:
                    field_value = field.get("value", {})
                    if isinstance(field_value, dict):
                        item_tenant_id = field_value.get("text", "").strip()
                    else:
                        item_tenant_id = str(field_value).strip()
                elif field.get("id") == status_field_id:
                    field_value = field.get("value", {})
                    if isinstance(field_value, dict):
                        item_status = field_value.get("text", "").strip()
                    else:
                        item_status = str(field_value).strip()
            
            # Count if tenant_id matches and status is pending
            if item_tenant_id == tenant_id and item_status and item_status.lower() == pending_label.lower():
                pending_count += 1
        
        return pending_count

