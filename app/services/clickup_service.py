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
                
                # Debug logging
                print(f"🔍 ClickUp API Key decrypted successfully")
                print(f"   Encrypted (first 20 chars): {self.api_key[:20]}...")
                print(f"   Decrypted (first 10 chars): {decrypted[:10] if decrypted else 'None'}...")
                print(f"   Decrypted length: {len(decrypted) if decrypted else 0}")
                
                if not decrypted:
                    raise ValueError("Decrypted API key is empty")
                
                # Validate API key format (ClickUp keys usually start with "pk_" but some formats may differ)
                if not decrypted.startswith("pk_"):
                    print(f"⚠️ Warning: ClickUp API key doesn't start with 'pk_'. Using provided format: {decrypted[:30]}...")
                    # Still return it - ClickUp may have different key formats
                
                return decrypted
            except Exception as e:
                print(f"❌ ClickUp API key decryption failed: {str(e)}")
                raise ValueError(f"Failed to decrypt ClickUp API key: {str(e)}")
        else:
            # Already decrypted or plain text
            print(f"🔍 ClickUp API Key appears to be already decrypted")
            return self.api_key

    def build_container_url(self, container_id: str, space_id: Optional[str] = None) -> str:
        """Build URL for ClickUp list"""
        # Use simple format - ClickUp lists can be accessed directly with list_id
        # Format: https://app.clickup.com/{list_id}
        return f"https://app.clickup.com/{container_id}"

    def _headers(self) -> Dict[str, str]:
        """Get API headers"""
        api_key = self.get_api_key()
        
        if not api_key:
            raise ValueError("ClickUp API key is missing or could not be decrypted")
        
        # Final validation - ClickUp API keys are usually long (40+ chars)
        if len(api_key) < 20:
            raise ValueError(f"ClickUp API key seems too short: {len(api_key)} chars. Minimum expected: 20 chars")
        
        # Log API key format for debugging (first and last few chars only)
        print(f"🔍 Using ClickUp API key (length: {len(api_key)}, format: {api_key[:5]}...{api_key[-5:]})")
        
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
                print(f"🔍 Auto-detecting ClickUp space...")
                
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
                
                print(f"✅ Found team: {team_name} (ID: {team_id})")
                
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
                
                print(f"✅ Auto-detected ClickUp space: {space_name} (ID: {space_id})")
                
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
            print(f"❌ ClickUp API 401 Unauthorized Error:")
            print(f"   URL: {url}")
            print(f"   Space ID used: {space_id}")
            print(f"   Response status: {response.status_code}")
            print(f"   Response body: {response.text[:500]}")
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
                print(f"🔍 Creating field {field_name} (type: {field_type})...")
                print(f"   Field data: {json.dumps(field_data, indent=2)}")
                create_response = requests.post(create_url, json=field_data, headers=self._headers(), timeout=20)
                if create_response.status_code == 200:
                    created_field = create_response.json()
                    field_id = created_field.get("id", "")
                    if field_id:
                        field_map[field_key] = field_id
                        print(f"✅ Created field {field_name} with ID: {field_id}")
                    else:
                        print(f"⚠️ Field {field_name} created but no ID returned: {create_response.text}")
                else:
                    print(f"❌ Failed to create field {field_name}:")
                    print(f"   Status: {create_response.status_code}")
                    print(f"   Response: {create_response.text}")
                    print(f"   Field data sent: {json.dumps(field_data, indent=2)}")
            else:
                # Use existing field
                existing_field_id = existing_names[field_name.lower()].get("id", "")
                if existing_field_id:
                    field_map[field_key] = existing_field_id
                    print(f"✅ Using existing field {field_name} with ID: {existing_field_id}")
                else:
                    print(f"⚠️ Existing field {field_name} found but has no ID")
        
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
        """Create a scheduled call task in ClickUp list"""
        url = f"{self.API_URL}/list/{container_id}/task"
        
        # Build custom fields (only if field_id exists and is not empty)
        custom_fields = []
        for key, field_id in field_map.items():
            # Skip if field_id is empty or missing
            if not field_id or field_id.strip() == "":
                print(f"⚠️ Skipping field {key} - field_id is empty")
                continue
            
            if key == "status":
                custom_fields.append({
                    "id": field_id,
                    "value": "Pending"
                })
            elif key == "email_sent":
                custom_fields.append({
                    "id": field_id,
                    "value": "No"
                })
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
                print(f"❌ ClickUp task creation failed:")
                print(f"   URL: {url}")
                print(f"   Status: {response.status_code}")
                print(f"   Response: {response.text[:500]}")
                print(f"   Payload: {json.dumps(payload, indent=2)}")
                
                try:
                    error_data = response.json()
                    error_msg = error_data.get("err", "") or error_data.get("error", "") or str(error_data)
                    print(f"   Error: {error_msg}")
                except:
                    pass
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            print(f"❌ Failed to create ClickUp task for {phone_number}: {error_msg}")
            raise ValueError(f"Failed to create ClickUp task: {error_msg}")
        except Exception as exc:
            print(f"❌ Failed to create ClickUp task for {phone_number}: {exc}")
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
        except Exception as exc:
            print(f"⚠️ Failed to update ClickUp task {item_id} status: {exc}")
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
        except Exception as exc:
            print(f"⚠️ Failed to update call_session_id for ClickUp task {item_id}: {exc}")
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
        """Delete tasks from ClickUp list that belong to a specific tenant"""
        tenant_field_id = field_map.get("tenant_id")
        if not tenant_field_id:
            raise ValueError("tenant_id field not found in field map")
        
        deleted = 0
        page = 0
        
        while True:
            # Fetch tasks from list
            url = f"{self.API_URL}/list/{container_id}/task"
            params = {
                "page": page,
                "limit": batch_size,
                "archived": "false"
            }
            
            try:
                response = requests.get(url, headers=self._headers(), params=params, timeout=20)
                response.raise_for_status()
                tasks = response.json().get("tasks", [])
            except Exception as exc:
                print(f"⚠️ Failed to fetch ClickUp tasks: {exc}")
                break
            
            if not tasks:
                break
            
            for task in tasks:
                # Get custom fields
                task_id = task.get("id", "")
                custom_fields = task.get("custom_fields", [])
                
                # Check tenant_id field
                item_tenant_id = None
                for field in custom_fields:
                    if field.get("id") == tenant_field_id:
                        item_tenant_id = field.get("value", "").strip()
                        break
                
                # Delete if tenant_id matches
                if item_tenant_id == tenant_id:
                    try:
                        delete_url = f"{self.API_URL}/task/{task_id}"
                        delete_response = requests.delete(delete_url, headers=self._headers(), timeout=20)
                        delete_response.raise_for_status()
                        deleted += 1
                        print(f"✅ Deleted ClickUp task {task_id} (tenant: {tenant_id})")
                    except Exception as exc:
                        print(f"⚠️ Failed to delete ClickUp task {task_id}: {exc}")
            
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
        """Count pending tasks from ClickUp list that belong to a specific tenant"""
        tenant_field_id = field_map.get("tenant_id")
        status_field_id = field_map.get("status")
        if not tenant_field_id or not status_field_id:
            raise ValueError("tenant_id or status field not found in field map")
        
        pending_count = 0
        page = 0
        
        while True:
            # Fetch tasks from list
            url = f"{self.API_URL}/list/{container_id}/task"
            params = {
                "page": page,
                "limit": batch_size,
                "archived": "false"
            }
            
            try:
                response = requests.get(url, headers=self._headers(), params=params, timeout=20)
                response.raise_for_status()
                tasks = response.json().get("tasks", [])
            except Exception as exc:
                print(f"⚠️ Failed to fetch ClickUp tasks: {exc}")
                break
            
            if not tasks:
                break
            
            for task in tasks:
                # Get custom fields
                custom_fields = task.get("custom_fields", [])
                
                # Check tenant_id and status fields
                item_tenant_id = None
                item_status = None
                for field in custom_fields:
                    if field.get("id") == tenant_field_id:
                        item_tenant_id = field.get("value", "").strip()
                    elif field.get("id") == status_field_id:
                        item_status = field.get("value", "").strip()
                
                # Count if tenant_id matches and status is pending
                if item_tenant_id == tenant_id and item_status and item_status.lower() == pending_label.lower():
                    pending_count += 1
            
            # Check if more pages
            if len(tasks) < batch_size:
                break
            page += 1
        
        return pending_count

