"""
Trello API Service for Scheduled Calls Integration
"""

import json
import re
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
        
        try:
            response = requests.post(url, params=params, timeout=20)
            response.raise_for_status()
            board_data = response.json()
            
            # Try multiple ID fields (shortLink is preferred, but id also works)
            board_id = board_data.get("shortLink", "") or board_data.get("id", "")
            if not board_id:
                raise ValueError(f"Trello API did not return board ID. Response: {board_data}")
            
            return {
                "id": board_id,
                "url": self.build_container_url(board_id),
            }
        except requests.exceptions.HTTPError as e:
            error_msg = f"Failed to create Trello board: "
            if e.response.status_code == 401:
                error_msg += "Authentication failed - check your API key and token."
            elif e.response.status_code == 403:
                error_msg += "Permission denied - API key/token may not have board creation access."
            else:
                try:
                    error_data = e.response.json()
                    error_msg += f"HTTP {e.response.status_code}: {error_data}"
                except:
                    error_msg += f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            raise ValueError(error_msg)
        except Exception as e:
            raise ValueError(f"Failed to create Trello board: {str(e)}")

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
        except Exception as e:
            print(f"⚠️ Failed to get Trello custom fields: {e}")
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
                print(f"✅ Found existing Trello field: {field_name} (ID: {field_map[field_key]})")
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
                        field_id = created_field.get("id", "")
                        if field_id:
                            field_map[field_key] = field_id
                            print(f"✅ Created Trello field: {field_name} (ID: {field_id})")
                        else:
                            print(f"⚠️ Created Trello field but no ID returned: {field_name}")
                    else:
                        print(f"⚠️ Failed to create Trello field {field_name}: HTTP {create_response.status_code} - {create_response.text[:200]}")
                except Exception as e:
                    print(f"⚠️ Failed to create Trello custom field {field_name}: {e}")
        
        print(f"📊 Trello field_map: {field_map}")
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
            field_update_errors = []
            for key, field_id in field_map.items():
                if not field_id:
                    continue
                
                update_url = f"{self.API_URL}/cards/{card_id}/customField/{field_id}/item"
                update_params = self._auth_params()
                
                try:
                    if key == "status":
                        # Set status to "Pending"
                        update_params.update({
                            "value": {"text": "Pending"}
                        })
                    elif key == "email_sent":
                        # Set email_sent to "No" by default
                        update_params.update({
                            "value": {"text": "No"}
                        })
                    elif key == "agent_id":
                        update_params.update({
                            "value": {"text": agent_id}
                        })
                    elif key == "call_time_utc":
                        update_params.update({
                            "value": {"text": call_time_utc}
                        })
                    elif key == "tenant_id":
                        update_params.update({
                            "value": {"text": tenant_id}
                        })
                    elif key == "user_id":
                        update_params.update({
                            "value": {"text": user_id}
                        })
                    elif key == "phone_number_id" and phone_number_id:
                        update_params.update({
                            "value": {"text": phone_number_id}
                        })
                    elif key == "batch_id" and batch_id:
                        update_params.update({
                            "value": {"text": batch_id}
                        })
                    elif key == "call_session_id":
                        # Leave blank initially (will be updated later when call is initiated)
                        update_params.update({
                            "value": {"text": ""}  # Blank initially
                        })
                    else:
                        continue  # Skip unknown fields
                    
                    response = requests.put(update_url, params=update_params, timeout=20)
                    response.raise_for_status()
                    print(f"✅ Updated Trello field {key} for card {card_id}")
                except Exception as e:
                    error_msg = f"Failed to update field {key}: {str(e)}"
                    field_update_errors.append(error_msg)
                    print(f"⚠️ {error_msg}")
            
            # If custom fields failed, add data to card description as fallback
            if field_update_errors or not field_map:
                print(f"⚠️ Some Trello custom fields failed or missing. Adding data to card description as fallback.")
                desc_lines = [f"Scheduled call at {call_time_utc}"]
                desc_lines.append(f"Agent ID: {agent_id}")
                desc_lines.append(f"User ID: {user_id}")
                desc_lines.append(f"Tenant ID: {tenant_id}")
                if phone_number_id:
                    desc_lines.append(f"Phone Number ID: {phone_number_id}")
                if batch_id:
                    desc_lines.append(f"Batch ID: {batch_id}")
                desc_lines.append(f"Status: Pending")
                desc_lines.append(f"Email Sent: No")
                
                # Update card description
                try:
                    update_desc_url = f"{self.API_URL}/cards/{card_id}"
                    update_desc_params = self._auth_params()
                    update_desc_params.update({
                        "desc": "\n".join(desc_lines)
                    })
                    desc_response = requests.put(update_desc_url, params=update_desc_params, timeout=20)
                    desc_response.raise_for_status()
                    print(f"✅ Updated Trello card description with field data")
                except Exception as e:
                    print(f"⚠️ Failed to update card description: {e}")
            
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

    def get_items_by_batch_id(
        self,
        container_id: str,
        batch_id: str,
        tenant_id: str,
        field_map: Dict[str, str],
        batch_size: int = 100
    ) -> List[Dict]:
        """
        Get all Trello cards with a specific batch_id and tenant_id.
        
        According to Trello API documentation:
        - Cards are fetched from board using GET /boards/{id}/cards
        - Custom fields are fetched per card using GET /cards/{id}/customFields
        - Falls back to description parsing if custom fields are not available
        
        Args:
            container_id: Trello board ID
            batch_id: Batch ID to filter by
            tenant_id: Tenant ID to filter by
            field_map: Field mapping dictionary (must include "batch_id" and "tenant_id")
            batch_size: Batch size for processing (not used for Trello, kept for compatibility)
            
        Returns:
            List of card dictionaries formatted with column_values for compatibility
            Each card dict includes: id, name, column_values (with call_session_id if available)
        """
        batch_field_id = field_map.get("batch_id")
        tenant_field_id = field_map.get("tenant_id")
        call_session_field_id = field_map.get("call_session_id")
        
        # If custom fields are missing, we'll use description parsing as fallback
        # This is common when Trello Power-Ups are not enabled
        if not batch_field_id or not tenant_field_id:
            print(f"⚠️ Warning: batch_id or tenant_id custom fields not found in field_map")
            print(f"   Field map: {field_map}")
            print(f"   Will use description parsing as fallback for batch_id and tenant_id")
            # Don't raise error - we'll parse from description instead
        
        print(f"🔍 Fetching Trello cards by batch_id:")
        print(f"   Board ID: {container_id}")
        print(f"   Batch ID: {batch_id}")
        print(f"   Tenant ID: {tenant_id}")
        print(f"   Batch field ID: {batch_field_id}")
        print(f"   Tenant field ID: {tenant_field_id}")
        
        items = []
        
        # Get all cards from board
        # Trello API: GET /boards/{id}/cards
        url = f"{self.API_URL}/boards/{container_id}/cards"
        params = self._auth_params()
        params.update({"filter": "all"})  # Get all cards including archived
        
        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            cards = response.json()
            print(f"📋 Fetched {len(cards)} cards from board")
        except Exception as exc:
            print(f"⚠️ Failed to fetch Trello cards: {exc}")
            return []
        
        for card in cards:
            card_id = card.get("id", "")
            card_name = card.get("name", "")
            card_desc = card.get("desc", "")
            
            # Get custom fields for this card
            # Trello API: GET /cards/{id}/customFields
            custom_fields_url = f"{self.API_URL}/card/{card_id}/customFields"
            custom_fields_params = self._auth_params()
            
            try:
                custom_fields_response = requests.get(custom_fields_url, params=custom_fields_params, timeout=20)
                custom_fields_response.raise_for_status()
                custom_fields = custom_fields_response.json()
            except Exception as exc:
                print(f"⚠️ Failed to get custom fields for card {card_id}: {exc}")
                custom_fields = []
            
            # Extract batch_id, tenant_id, and call_session_id
            item_batch_id = None
            item_tenant_id = None
            item_call_session_id = None
            
            # First try custom fields (preferred method) - only if field IDs are available
            if batch_field_id or tenant_field_id or call_session_field_id:
                for field in custom_fields:
                    field_id = field.get("id", "")
                    field_value = field.get("value", {})
                    
                    if batch_field_id and field_id == batch_field_id:
                        # Trello custom field values are objects with "text" property for text fields
                        if isinstance(field_value, dict):
                            item_batch_id = field_value.get("text", "").strip()
                        else:
                            item_batch_id = str(field_value).strip() if field_value else None
                    elif tenant_field_id and field_id == tenant_field_id:
                        if isinstance(field_value, dict):
                            item_tenant_id = field_value.get("text", "").strip()
                        else:
                            item_tenant_id = str(field_value).strip() if field_value else None
                    elif call_session_field_id and field_id == call_session_field_id:
                        if isinstance(field_value, dict):
                            item_call_session_id = field_value.get("text", "").strip()
                        else:
                            item_call_session_id = str(field_value).strip() if field_value else None
            
            # Fallback to description parsing if custom fields not found
            if not item_batch_id and card_desc:
                batch_match = re.search(r'Batch ID:\s*([^\n]+)', card_desc, re.IGNORECASE)
                if batch_match:
                    item_batch_id = batch_match.group(1).strip()
            
            if not item_tenant_id and card_desc:
                tenant_match = re.search(r'Tenant ID:\s*([^\n]+)', card_desc, re.IGNORECASE)
                if tenant_match:
                    item_tenant_id = tenant_match.group(1).strip()
            
            if not item_call_session_id and card_desc:
                session_match = re.search(r'Call Session ID:\s*([^\n]+)', card_desc, re.IGNORECASE)
                if session_match:
                    item_call_session_id = session_match.group(1).strip()
            
            # Match by batch_id and tenant_id
            if item_batch_id == batch_id and item_tenant_id == tenant_id:
                # Format item similar to Monday.com/ClickUp format for compatibility
                # This ensures the batch analysis endpoint can process Trello items the same way
                formatted_item = {
                    "id": card_id,
                    "name": card_name,
                    "column_values": []  # Format for compatibility with existing code
                }
                
                # Add call_session_id to column_values format (required by batch analysis endpoint)
                if item_call_session_id and call_session_field_id:
                    formatted_item["column_values"].append({
                        "id": call_session_field_id,
                        "text": item_call_session_id
                    })
                
                items.append(formatted_item)
                print(f"✅ Matched card {card_id} ({card_name}) - Batch: {item_batch_id}, Tenant: {item_tenant_id}")
        
        print(f"✅ Found {len(items)} cards matching batch_id={batch_id} and tenant_id={tenant_id}")
        return items

    def update_item_email_sent(
        self,
        container_id: str,
        item_id: str,
        field_map: Dict[str, str],
    ) -> Optional[dict]:
        """
        Update Email Sent status to "Yes" for a Trello card.
        Tries custom field first, falls back to description update.
        
        Args:
            container_id: Trello board ID
            item_id: Trello card ID
            field_map: Field mapping dictionary
            
        Returns:
            Updated card data if successful, None otherwise
        """
        email_sent_field_id = field_map.get("email_sent")
        updated = False
        
        # Try custom field update first (if available)
        if email_sent_field_id:
            url = f"{self.API_URL}/cards/{item_id}/customField/{email_sent_field_id}/item"
            params = self._auth_params()
            params.update({
                "value": {"text": "Yes"}
            })
            
            try:
                response = requests.put(url, params=params, timeout=20)
                response.raise_for_status()
                print(f"✅ Updated Email Sent custom field to 'Yes' for card {item_id}")
                updated = True
            except Exception as exc:
                print(f"⚠️ Failed to update Email Sent custom field for card {item_id}: {exc}")
        
        # Fallback to description update (always try, even if custom field succeeded)
        # This ensures description is also updated for consistency
        try:
            # Get current card description
            get_url = f"{self.API_URL}/cards/{item_id}"
            get_params = self._auth_params()
            get_response = requests.get(get_url, params=get_params, timeout=20)
            get_response.raise_for_status()
            card_data = get_response.json()
            description = card_data.get("desc", "")
            
            # Update description
            if description:
                # Replace Email Sent: No with Email Sent: Yes
                if 'Email Sent: No' in description:
                    description = description.replace('Email Sent: No', 'Email Sent: Yes')
                elif re.search(r'Email Sent:\s*(Yes|No)', description, re.IGNORECASE):
                    description = re.sub(r'Email Sent:\s*(Yes|No)', 'Email Sent: Yes', description, flags=re.IGNORECASE)
                else:
                    description = description + ('\n' if not description.endswith('\n') else '') + 'Email Sent: Yes'
            else:
                description = 'Email Sent: Yes'
            
            # Update card description
            update_url = f"{self.API_URL}/cards/{item_id}"
            update_params = self._auth_params()
            update_params.update({
                "desc": description
            })
            update_response = requests.put(update_url, params=update_params, timeout=20)
            update_response.raise_for_status()
            print(f"✅ Updated Email Sent in description to 'Yes' for card {item_id}")
            updated = True
            return update_response.json()
        except Exception as exc:
            print(f"⚠️ Failed to update Email Sent in description for card {item_id}: {exc}")
            if updated:
                # Custom field was updated, so return success
                return {"id": item_id}
            return None

    def update_items_email_sent(
        self,
        container_id: str,
        item_ids: List[str],
        field_map: Dict[str, str],
    ) -> int:
        """
        Update Email Sent status to "Yes" for multiple Trello cards.
        
        Args:
            container_id: Trello board ID
            item_ids: List of Trello card IDs
            field_map: Field mapping dictionary
            
        Returns:
            Number of successfully updated cards
        """
        if not item_ids:
            return 0
        
        updated_count = 0
        for item_id in item_ids:
            result = self.update_item_email_sent(container_id, item_id, field_map)
            if result:
                updated_count += 1
        
        return updated_count

