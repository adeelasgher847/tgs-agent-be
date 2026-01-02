"""
ClickUp API Service for Scheduled Calls Integration
"""

import json
from typing import Dict, List, Optional
import requests
from app.services.base_crm_service import BaseCRMService
from app.core.security import decrypt_api_key


class ClickUpService(BaseCRMService):
    """Service for interacting with ClickUp API"""

    API_URL = "https://api.clickup.com/api/v2"
    REQUIRED_FIELDS = [
        {
            "key": "status",
            "title": "Status",
            "type": "dropdown",
            "defaults": {"options": ["Pending", "Called", "Failed"]},
        },
        {"key": "agent_id", "title": "Agent ID", "type": "short_text"},
        {"key": "call_time_utc", "title": "Call Time UTC", "type": "short_text"},
        {"key": "tenant_id", "title": "Tenant ID", "type": "short_text"},
        {"key": "user_id", "title": "User ID", "type": "short_text"},
        {"key": "batch_id", "title": "Batch ID", "type": "short_text"},
        {"key": "call_session_id", "title": "Call Session ID", "type": "short_text"},
        {"key": "phone_number_id", "title": "Phone Number ID", "type": "short_text"},
        {
            "key": "email_sent",
            "title": "Email Sent",
            "type": "dropdown",
            "defaults": {"options": ["No", "Yes"]},
        },
    ]

    def __init__(self, api_key: str):
        """Initialize ClickUp service with API key"""
        self.api_key = api_key

    def get_api_key(self) -> str:
        """Get decrypted API key"""
        if not self.api_key or self.api_key.strip() == "":
            raise ValueError("ClickUp API key is not configured. Please complete OAuth authorization first.")
        
        # Check if encrypted (JWT format)
        if self.api_key.startswith("eyJ"):
            try:
                decrypted = decrypt_api_key(self.api_key)
                
                if not decrypted:
                    raise ValueError("Decrypted API key is empty")
                
                # Validate API key format (ClickUp keys usually start with "pk_" but some formats may differ)
                if not decrypted.startswith("pk_"):
                    # Still return it - ClickUp may have different key formats
                    pass
                
                return decrypted
            except Exception as e:
                raise ValueError(f"Failed to decrypt ClickUp API key: {str(e)}")
        else:
            # Already decrypted or plain text
            return self.api_key

    def build_container_url(self, container_id: str, space_id: Optional[str] = None) -> str:
        """Build URL for ClickUp list"""
        # Use simple format - ClickUp lists can be accessed directly with list_id
        # Format: https://app.clickup.com/{list_id}
        return f"https://app.clickup.com/{container_id}"

    def get_list_url(self, list_id: str) -> str:
        """
        Get proper ClickUp list URL by fetching list details from API.
        Fetches list details from ClickUp API to get proper URL.
        """
        try:
            # Fetch list details from ClickUp API
            url = f"{self.API_URL}/list/{list_id}"
            response = requests.get(url, headers=self._headers(), timeout=20)
            response.raise_for_status()
            list_data = response.json()
            
            # ClickUp API returns 'url' field in list response
            if list_data.get("url"):
                return list_data["url"]
            
            # Fallback: Build URL from list_id
            return self.build_container_url(list_id)
            
        except Exception:
            # If API call fails, fallback to basic URL
            return self.build_container_url(list_id)

    def _headers(self) -> Dict[str, str]:
        """Get API headers"""
        api_key = self.get_api_key()
        
        if not api_key:
            raise ValueError("ClickUp API key is missing or could not be decrypted")
        
        # Final validation - ClickUp API keys are usually long (40+ chars)
        if len(api_key) < 20:
            raise ValueError(f"ClickUp API key seems too short: {len(api_key)} chars. Minimum expected: 20 chars")
        
        # ClickUp OAuth access tokens require "Bearer" prefix
        # Personal API tokens (pk_*) don't need Bearer prefix
        if api_key.startswith("pk_"):
            # Personal API token format (old format)
            auth_header = api_key
        else:
            # OAuth access token - requires Bearer prefix
            auth_header = f"Bearer {api_key}"
        
        return {
            "Authorization": auth_header,
            "Content-Type": "application/json",
        }

    def create_container(self, container_name: str, space_id: Optional[str] = None, folder_id: Optional[str] = None) -> Dict[str, str]:
        """
        Create a ClickUp list for scheduled calls.
        Automatically gets default space if space_id not provided (like Monday.com).
        
        Args:
            container_name: Name for the list
            space_id: ClickUp space ID (optional - will auto-detect if not provided)
            folder_id: ClickUp folder ID (optional)
        """
        # If space_id not provided, auto-detect from team (like Monday.com auto-detects workspace)
        if not space_id:
            try:
                # Get team (workspace) - API key identifies the team
                team_url = f"{self.API_URL}/team"
                team_response = requests.get(team_url, headers=self._headers(), timeout=20)
                team_response.raise_for_status()
                teams_data = team_response.json()
                teams = teams_data.get("teams", [])
                
                if not teams:
                    raise ValueError("No teams found for this ClickUp API key")
                
                # Use first team
                team_id = teams[0].get("id", "")
                team_name = teams[0].get("name", "Unknown")
                if not team_id:
                    raise ValueError("Could not get team ID")
                
                # Get spaces for this team
                spaces_url = f"{self.API_URL}/team/{team_id}/space"
                spaces_response = requests.get(spaces_url, headers=self._headers(), timeout=20)
                spaces_response.raise_for_status()
                spaces_data = spaces_response.json()
                spaces = spaces_data.get("spaces", [])
                
                if not spaces:
                    raise ValueError(f"No spaces found in team {team_name}")
                
                # Use first space
                space_id = spaces[0].get("id", "")
                space_name = spaces[0].get("name", "Unknown")
                if not space_id:
                    raise ValueError("Could not get space ID")
                
            except requests.exceptions.HTTPError as e:
                error_msg = f"Failed to auto-detect ClickUp space. Please provide space_id in additional_config."
                if e.response.status_code == 401:
                    error_msg += " Authentication failed - check your API key."
                elif e.response.status_code == 403:
                    error_msg += " Permission denied - API key may not have access to teams/spaces."
                else:
                    error_msg += f" HTTP {e.response.status_code}: {e.response.text[:200]}"
                raise ValueError(error_msg)
            except Exception as e:
                raise ValueError(f"Failed to auto-detect ClickUp space. Please provide space_id in additional_config. Error: {str(e)}")
        
        # Create list in the space
        url = f"{self.API_URL}/space/{space_id}/list"
        if folder_id:
            url = f"{self.API_URL}/folder/{folder_id}/list"
        
        payload = {
            "name": container_name,
            "content": "Scheduled Calls List",
        }
        
        response = requests.post(url, json=payload, headers=self._headers(), timeout=20)
        
        # Better error logging for 401
        if response.status_code == 401:
            error_msg = "ClickUp API authentication failed. "
            error_msg += "Possible causes: Invalid/expired token, insufficient scopes (need 'read' and 'write'), or user lacks permission to create lists in this space. "
            error_msg += f"Response: {response.text[:200]}"
            raise ValueError(error_msg)
        
        response.raise_for_status()
        data = response.json()
        
        list_id = data.get("id", "")
        return {
            "id": list_id,
            "url": self.build_container_url(list_id, space_id=space_id),
        }

    def ensure_required_fields(self, container_id: str) -> Dict[str, str]:
        """
        Ensure the ClickUp list has all required custom fields.
        Creates missing fields if needed.
        """
        # Get existing custom fields
        url = f"{self.API_URL}/list/{container_id}/field"
        response = requests.get(url, headers=self._headers(), timeout=20)
        response.raise_for_status()
        existing_fields = response.json().get("fields", [])
        
        # Map existing fields by name
        field_map = {}
        existing_names = {f.get("name", "").lower(): f for f in existing_fields}
        
        # Create missing fields
        for field_def in self.REQUIRED_FIELDS:
            field_name = field_def["title"]
            field_key = field_def["key"]
            
            if field_name.lower() not in existing_names:
                # Create custom field
                # Fix dropdown type - ClickUp uses "drop_down" not "dropdown"
                field_type = field_def["type"]
                if field_type == "dropdown":
                    field_type = "drop_down"  # ClickUp API format
                
                field_data = {
                    "name": field_name,
                    "type": field_type,
                }
                
                if "defaults" in field_def:
                    # For dropdown, type_config should have options array with objects
                    if field_type == "drop_down":
                        # ClickUp requires options as array of objects with "name" key
                        options_list = field_def["defaults"].get("options", [])
                        field_data["type_config"] = {
                            "options": [{"name": opt} for opt in options_list]
                        }
                    else:
                        field_data["type_config"] = field_def["defaults"]
                
                create_url = f"{self.API_URL}/list/{container_id}/field"
                create_response = requests.post(create_url, json=field_data, headers=self._headers(), timeout=20)
                if create_response.status_code == 200:
                    created_field = create_response.json()
                    # ClickUp returns field object under "field" key, not top-level
                    field_obj = created_field.get("field") or created_field
                    field_id = field_obj.get("id", "")
                    if field_id:
                        field_map[field_key] = field_id
            else:
                # Use existing field
                existing_field_id = existing_names[field_name.lower()].get("id", "")
                if existing_field_id:
                    field_map[field_key] = existing_field_id
        
        return field_map

    def _get_dropdown_option_uuid(self, container_id: str, field_id: str, option_name: str) -> Optional[str]:
        """
        Get the UUID of a dropdown option by its name.
        ClickUp dropdown fields require option UUID, not the option name string.
        """
        try:
            # Fetch all fields from the list (ClickUp doesn't have individual field endpoint)
            url = f"{self.API_URL}/list/{container_id}/field"
            response = requests.get(url, headers=self._headers(), timeout=20)
            response.raise_for_status()
            all_fields = response.json().get("fields", [])
            
            # Find the specific field by ID
            field_data = None
            for field in all_fields:
                if field.get("id", "") == field_id:
                    field_data = field
                    break
            
            if not field_data:
                return None
            
            # Get options from type_config
            type_config = field_data.get("type_config", {})
            options = type_config.get("options", [])
            
            if not options:
                return None
            
            # Find matching option by name (case-insensitive)
            for option in options:
                option_label = option.get("label", "") or option.get("name", "")
                if option_label.lower() == option_name.lower():
                    # Return the option ID (UUID)
                    option_id = option.get("id", "") or option.get("uuid", "")
                    if option_id:
                        return option_id
                    else:
                        # Sometimes option might have index instead of UUID
                        option_index = option.get("orderindex", None)
                        if option_index is not None:
                            return str(option_index)
            
            return None
        except Exception:
            return None

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
        """Create a scheduled call task in ClickUp list"""
        url = f"{self.API_URL}/list/{container_id}/task"
        
        # Build custom fields (only if field_id exists and is not empty)
        custom_fields = []
        for key, field_id in field_map.items():
            # Skip if field_id is empty or missing
            if not field_id or field_id.strip() == "":
                continue
            
            if key == "status":
                # For dropdown fields, we need to get the option UUID
                option_uuid = self._get_dropdown_option_uuid(container_id, field_id, "Pending")
                if option_uuid:
                    custom_fields.append({
                        "id": field_id,
                        "value": option_uuid
                    })
                else:
                    # Don't add the field if UUID not found - it will cause 400 error
                    pass
            elif key == "email_sent":
                # For dropdown fields, we need to get the option UUID
                option_uuid = self._get_dropdown_option_uuid(container_id, field_id, "No")
                if option_uuid:
                    custom_fields.append({
                        "id": field_id,
                        "value": option_uuid
                    })
                else:
                    # Don't add the field if UUID not found - it will cause 400 error
                    pass
            elif key == "agent_id":
                custom_fields.append({
                    "id": field_id,
                    "value": agent_id
                })
            elif key == "call_time_utc":
                custom_fields.append({
                    "id": field_id,
                    "value": call_time_utc
                })
            elif key == "tenant_id":
                custom_fields.append({
                    "id": field_id,
                    "value": tenant_id
                })
            elif key == "user_id":
                custom_fields.append({
                    "id": field_id,
                    "value": user_id
                })
            elif key == "batch_id" and batch_id:
                custom_fields.append({
                    "id": field_id,
                    "value": batch_id
                })
            elif key == "phone_number_id" and phone_number_id:
                custom_fields.append({
                    "id": field_id,
                    "value": phone_number_id
                })
            elif key == "call_session_id":
                # Leave blank initially (will be updated later when call is initiated)
                custom_fields.append({
                    "id": field_id,
                    "value": ""  # Blank initially
                })
        
        payload = {
            "name": phone_number,
            "description": f"Scheduled call for {phone_number}",
            "custom_fields": custom_fields,
        }
        
        try:
            response = requests.post(url, json=payload, headers=self._headers(), timeout=20)
            
            # Better error logging
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("err", "") or error_data.get("error", "") or str(error_data)
                except:
                    error_msg = response.text[:200]
            
            response.raise_for_status()
            created_task = response.json()
            
            # Verify tenant_id was saved by fetching the task immediately
            task_id = created_task.get("id", "")
            if task_id:
                try:
                    verify_url = f"{self.API_URL}/task/{task_id}"
                    verify_response = requests.get(verify_url, headers=self._headers(), timeout=20)
                    verify_response.raise_for_status()
                    verified_task = verify_response.json()
                    verified_fields = verified_task.get("custom_fields", [])
                    tenant_field_verified = next((f for f in verified_fields if f.get("id") == field_map.get("tenant_id")), None)
                except Exception:
                    pass
            
            return created_task
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            raise ValueError(f"Failed to create ClickUp task: {error_msg}")
        except Exception as exc:
            raise ValueError(f"Failed to create ClickUp task: {str(exc)}")

    def update_item_status(
        self,
        container_id: str,
        item_id: str,
        status: str,
        field_map: Dict[str, str],
    ) -> Optional[dict]:
        """Update task status in ClickUp"""
        status_field_id = field_map.get("status")
        if not status_field_id:
            return None
        
        url = f"{self.API_URL}/task/{item_id}/field/{status_field_id}"
        payload = {"value": status}
        
        try:
            response = requests.post(url, json=payload, headers=self._headers(), timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

    def update_item_call_session_id(
        self,
        container_id: str,
        item_id: str,
        call_session_id: str,
        field_map: Dict[str, str],
    ) -> Optional[dict]:
        """Update call_session_id field for a ClickUp task"""
        session_field_id = field_map.get("call_session_id")
        if not session_field_id:
            return None
        
        url = f"{self.API_URL}/task/{item_id}/field/{session_field_id}"
        payload = {"value": call_session_id}
        
        try:
            response = requests.post(url, json=payload, headers=self._headers(), timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

    def update_item_email_sent(
        self,
        container_id: str,
        item_id: str,
        field_map: Dict[str, str],
    ) -> Optional[dict]:
        """Update Email Sent field to 'Yes' for a ClickUp task"""
        email_sent_field_id = field_map.get("email_sent")
        if not email_sent_field_id:
            return None
        
        # Get UUID for "Yes" option
        yes_uuid = self._get_dropdown_option_uuid(container_id, email_sent_field_id, "Yes")
        if not yes_uuid:
            return None
        
        url = f"{self.API_URL}/task/{item_id}/field/{email_sent_field_id}"
        payload = {"value": yes_uuid}
        
        try:
            response = requests.post(url, json=payload, headers=self._headers(), timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

    def update_items_email_sent(
        self,
        container_id: str,
        item_ids: List[str],
        field_map: Dict[str, str],
    ) -> int:
        """Update Email Sent field to 'Yes' for multiple ClickUp tasks"""
        if not item_ids:
            return 0
        
        updated_count = 0
        for item_id in item_ids:
            result = self.update_item_email_sent(container_id, item_id, field_map)
            if result:
                updated_count += 1
        
        return updated_count

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
        """Delete tasks from ClickUp list that belong to a specific tenant"""
        tenant_field_id = field_map.get("tenant_id")
        if not tenant_field_id:
            raise ValueError("tenant_id field not found in field map")
        
        deleted = 0
        page = 0
        
        while True:
            # Fetch tasks from list
            # Note: ClickUp API may not return full custom field data in list endpoint
            # We'll fetch basic task info first, then get detailed custom fields per task if needed
            url = f"{self.API_URL}/list/{container_id}/task"
            params = {
                "page": page,
                "limit": batch_size,
                "archived": "false",
                "include_closed": "false"
            }
            
            try:
                response = requests.get(url, headers=self._headers(), params=params, timeout=20)
                response.raise_for_status()
                tasks = response.json().get("tasks", [])
            except Exception:
                break
            
            if not tasks:
                break
            
            for task in tasks:
                # Get task basic info
                task_id = task.get("id", "")
                task_name = task.get("name", "")
                custom_fields = task.get("custom_fields", [])
                
                # ClickUp list endpoint doesn't return custom field values properly
                # Always fetch individual task details to get actual custom field values
                try:
                    task_detail_url = f"{self.API_URL}/task/{task_id}"
                    task_detail_response = requests.get(task_detail_url, headers=self._headers(), timeout=20)
                    task_detail_response.raise_for_status()
                    task_detail = task_detail_response.json()
                    custom_fields = task_detail.get("custom_fields", [])
                except Exception:
                    custom_fields = []
                
                # Check tenant_id field
                item_tenant_id = None
                tenant_field_found = False
                for field in custom_fields:
                    field_id = field.get("id", "")
                    if field_id == tenant_field_id:
                        tenant_field_found = True
                        # ClickUp custom field value can be in different formats
                        # For short_text fields, value is directly in "value" field
                        # But sometimes it might be empty string or None if not set
                        field_value = field.get("value")
                        
                        # Handle different value formats
                        # ClickUp short_text fields: value is a string or None
                        if field_value is None:
                            # Value is None, meaning it wasn't set when task was created
                            item_tenant_id = None
                        elif isinstance(field_value, str):
                            item_tenant_id = field_value.strip() if field_value.strip() else None
                        elif isinstance(field_value, dict):
                            # Sometimes value is nested in a dict (for other field types)
                            item_tenant_id = field_value.get("value", "") or field_value.get("text", "")
                            if item_tenant_id:
                                item_tenant_id = str(item_tenant_id).strip()
                        else:
                            item_tenant_id = str(field_value).strip() if field_value else None
                        break
                
                # Delete if tenant_id matches
                if item_tenant_id == tenant_id:
                    try:
                        delete_url = f"{self.API_URL}/task/{task_id}"
                        delete_response = requests.delete(delete_url, headers=self._headers(), timeout=20)
                        delete_response.raise_for_status()
                        deleted += 1
                    except Exception:
                        pass
            
            # Check if more pages
            if len(tasks) < batch_size:
                break
            page += 1
        
        return deleted

    def count_pending_items_for_tenant(
        self,
        container_id: str,
        tenant_id: str,
        field_map: Dict[str, str],
        pending_label: str = "Pending",
        batch_size: int = 100
    ) -> int:
        """
        Count pending tasks from ClickUp list that belong to a specific tenant.
        Fetches individual task details to get actual custom field values since
        the list endpoint doesn't return full custom field data.
        """
        tenant_field_id = field_map.get("tenant_id")
        status_field_id = field_map.get("status")
        if not tenant_field_id or not status_field_id:
            raise ValueError("tenant_id or status field not found in field map")
        
        pending_count = 0
        page = 0
        
        # Get the UUID for "Pending" option (status is a dropdown field)
        pending_option_uuid = None
        try:
            pending_option_uuid = self._get_dropdown_option_uuid(container_id, status_field_id, pending_label)
        except Exception:
            # Continue anyway - we'll try to match by name if UUID not found
            pass
        
        while True:
            # Fetch tasks from list
            url = f"{self.API_URL}/list/{container_id}/task"
            params = {
                "page": page,
                "limit": batch_size,
                "archived": "false",
                "include_closed": "false"
            }
            
            try:
                response = requests.get(url, headers=self._headers(), params=params, timeout=20)
                response.raise_for_status()
                tasks = response.json().get("tasks", [])
            except Exception:
                break
            
            if not tasks:
                break
            
            for task in tasks:
                task_id = task.get("id", "")
                task_name = task.get("name", "")
                
                # ClickUp list endpoint doesn't return full custom field values
                # Fetch individual task details to get actual custom field values
                try:
                    task_detail_url = f"{self.API_URL}/task/{task_id}"
                    task_detail_response = requests.get(task_detail_url, headers=self._headers(), timeout=20)
                    task_detail_response.raise_for_status()
                    task_detail = task_detail_response.json()
                    custom_fields = task_detail.get("custom_fields", [])
                except Exception:
                    # Fallback to list endpoint custom fields (may not have values)
                    custom_fields = task.get("custom_fields", [])
                
                # Check tenant_id and status fields
                item_tenant_id = None
                item_status = None
                item_status_name = None
                item_status_uuid = None
                
                # Check if status field exists in custom_fields
                status_field_found = False
                for field in custom_fields:
                    if field.get("id") == status_field_id:
                        status_field_found = True
                        break
                
                for field in custom_fields:
                    field_id = field.get("id", "")
                    field_value = field.get("value")
                    field_type = field.get("type", "")
                    
                    if field_id == tenant_field_id:
                        # Handle different value formats for tenant_id (short_text field)
                        if field_value is None:
                            item_tenant_id = None
                        elif isinstance(field_value, str):
                            item_tenant_id = field_value.strip() if field_value.strip() else None
                        elif isinstance(field_value, dict):
                            item_tenant_id = str(field_value.get("value", "") or field_value.get("text", "")).strip()
                        else:
                            item_tenant_id = str(field_value).strip() if field_value else None
                    elif field_id == status_field_id:
                        # Status is a dropdown field
                        # ClickUp dropdown fields return value as INTEGER (orderindex), not UUID!
                        # value: 0 = orderindex 0 = "Pending"
                        # value: 1 = orderindex 1 = "Called"
                        # value: 2 = orderindex 2 = "Failed"
                        
                        if field_value is None:
                            item_status = None
                            item_status_uuid = None
                            item_status_name = None
                        elif isinstance(field_value, (int, float)):
                            # Value is orderindex (integer)
                            orderindex = int(field_value)
                            
                            # Get the option from type_config.options by orderindex
                            type_config = field.get("type_config", {})
                            options = type_config.get("options", [])
                            
                            # Find option with matching orderindex
                            for option in options:
                                if option.get("orderindex") == orderindex:
                                    item_status_uuid = option.get("id", "")
                                    item_status_name = option.get("name", "") or option.get("label", "")
                                    item_status = item_status_uuid or item_status_name
                                    break
                        elif isinstance(field_value, str):
                            # Could be UUID string (legacy format)
                            item_status = field_value.strip()
                            item_status_uuid = item_status
                        elif isinstance(field_value, dict):
                            # Dropdown value is an object (alternative format)
                            item_status_uuid = field_value.get("id", "") or field_value.get("uuid", "")
                            item_status_name = field_value.get("name", "") or field_value.get("label", "")
                            item_status = item_status_uuid or item_status_name
                        else:
                            item_status = str(field_value).strip() if field_value else None
                            item_status_uuid = item_status
                
                # Check if tenant_id matches
                if item_tenant_id != tenant_id:
                    continue
                
                # Check if status is pending
                # Try multiple ways to match:
                # 1. Match by UUID (if we got it)
                # 2. Match by option name/label
                # 3. Match by string comparison (fallback)
                is_pending = False
                if pending_option_uuid:
                    # Try UUID match first (most reliable)
                    if item_status_uuid and str(item_status_uuid).strip() == str(pending_option_uuid).strip():
                        is_pending = True
                    elif item_status and str(item_status).strip() == str(pending_option_uuid).strip():
                        is_pending = True
                
                if not is_pending and item_status_name:
                    # Try name match
                    if item_status_name.lower() == pending_label.lower():
                        is_pending = True
                
                if not is_pending and item_status:
                    # Try string comparison (fallback)
                    if isinstance(item_status, str) and item_status.lower() == pending_label.lower():
                        is_pending = True
                
                if is_pending:
                    pending_count += 1
            
            # Check if more pages
            if len(tasks) < batch_size:
                break
            page += 1
        
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
        Fetch all tasks from a list with specific batch_id and tenant_id.
        
        Args:
            container_id: ClickUp list ID
            batch_id: Batch ID to filter by
            tenant_id: Tenant ID to filter by (UUID string)
            field_map: Field mapping dictionary (must include "batch_id" and "tenant_id")
            batch_size: Number of tasks to fetch per batch
            
        Returns:
            List of tasks matching the batch_id and tenant_id
            Each task dict includes: id, name, custom_fields (with values)
        """
        batch_field_id = field_map.get("batch_id")
        tenant_field_id = field_map.get("tenant_id")
        call_session_field_id = field_map.get("call_session_id")
        
        if not batch_field_id or not tenant_field_id:
            raise ValueError("batch_id or tenant_id field not found in field map")
        
        items = []
        page = 0
        
        while True:
            # Fetch tasks from list
            url = f"{self.API_URL}/list/{container_id}/task"
            params = {
                "page": page,
                "limit": batch_size,
                "archived": "false",
                "include_closed": "false"
            }
            
            try:
                response = requests.get(url, headers=self._headers(), params=params, timeout=20)
                response.raise_for_status()
                tasks = response.json().get("tasks", [])
            except Exception:
                break
            
            if not tasks:
                break
            
            for task in tasks:
                task_id = task.get("id", "")
                task_name = task.get("name", "")
                
                # ClickUp list endpoint doesn't return full custom field values
                # Fetch individual task details to get actual custom field values
                try:
                    task_detail_url = f"{self.API_URL}/task/{task_id}"
                    task_detail_response = requests.get(task_detail_url, headers=self._headers(), timeout=20)
                    task_detail_response.raise_for_status()
                    task_detail = task_detail_response.json()
                    custom_fields = task_detail.get("custom_fields", [])
                except Exception:
                    custom_fields = task.get("custom_fields", [])
                
                # Check if task belongs to this batch and tenant
                item_batch_id = None
                item_tenant_id = None
                item_call_session_id = None
                
                for field in custom_fields:
                    field_id = field.get("id", "")
                    field_value = field.get("value", "")
                    
                    if field_id == batch_field_id:
                        item_batch_id = str(field_value).strip() if field_value else None
                    elif field_id == tenant_field_id:
                        item_tenant_id = str(field_value).strip() if field_value else None
                    elif field_id == call_session_field_id and call_session_field_id:
                        item_call_session_id = str(field_value).strip() if field_value else None
                
                # Match by batch_id and tenant_id
                if item_batch_id == batch_id and item_tenant_id == tenant_id:
                    # Format task similar to Monday.com item format for consistency
                    item = {
                        "id": task_id,
                        "name": task_name,
                        "custom_fields": custom_fields,
                        # Add column_values format for compatibility
                        "column_values": [
                            {
                                "id": field.get("id"),
                                "text": str(field.get("value", "")),
                                "value": field.get("value")
                            }
                            for field in custom_fields
                        ]
                    }
                    items.append(item)
            
            # Check if more pages
            if len(tasks) < batch_size:
                break
            page += 1
        
        return items

