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
                    # ClickUp returns field object under "field" key, not top-level
                    field_obj = created_field.get("field") or created_field
                    field_id = field_obj.get("id", "")
                    if field_id:
                        field_map[field_key] = field_id
                        print(f"✅ Created field {field_name} with ID: {field_id}")
                    else:
                        print(f"⚠️ Field {field_name} created but no ID returned: {create_response.text}")
                        print(f"   Response structure: {json.dumps(created_field, indent=2)[:500]}")
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
                print(f"   ⚠️ Field with ID {field_id} not found in list")
                return None
            
            # Get options from type_config
            type_config = field_data.get("type_config", {})
            options = type_config.get("options", [])
            
            if not options:
                print(f"   ⚠️ No options found in dropdown field {field_data.get('name', field_id)}")
                return None
            
            # Find matching option by name (case-insensitive)
            for option in options:
                option_label = option.get("label", "") or option.get("name", "")
                if option_label.lower() == option_name.lower():
                    # Return the option ID (UUID)
                    option_id = option.get("id", "") or option.get("uuid", "")
                    if option_id:
                        print(f"   ✅ Found dropdown option '{option_name}' with UUID: {option_id}")
                        return option_id
                    else:
                        # Sometimes option might have index instead of UUID
                        option_index = option.get("orderindex", None)
                        if option_index is not None:
                            print(f"   ✅ Found dropdown option '{option_name}' with index: {option_index}")
                            return str(option_index)
            
            print(f"   ⚠️ Dropdown option '{option_name}' not found in field options")
            print(f"   Field name: {field_data.get('name', 'Unknown')}")
            print(f"   Available options: {[opt.get('label', opt.get('name', '')) for opt in options]}")
            return None
        except Exception as exc:
            print(f"   ⚠️ Failed to get dropdown option UUID for '{option_name}': {exc}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
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
                print(f"⚠️ Skipping field {key} - field_id is empty")
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
                    print(f"❌ ERROR: Could not find 'Pending' option for Status field. Skipping this field.")
                    # Don't add the field if UUID not found - it will cause 400 error
            elif key == "email_sent":
                # For dropdown fields, we need to get the option UUID
                option_uuid = self._get_dropdown_option_uuid(container_id, field_id, "No")
                if option_uuid:
                    custom_fields.append({
                        "id": field_id,
                        "value": option_uuid
                    })
                else:
                    print(f"❌ ERROR: Could not find 'No' option for Email Sent field. Skipping this field.")
                    # Don't add the field if UUID not found - it will cause 400 error
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
        
        # Debug: Log the payload to verify tenant_id is being sent
        print(f"🔍 Creating ClickUp task with payload:")
        print(f"   Task name: {phone_number}")
        print(f"   Custom fields count: {len(custom_fields)}")
        tenant_field_in_payload = next((f for f in custom_fields if f.get("id") == field_map.get("tenant_id")), None)
        if tenant_field_in_payload:
            print(f"   ✅ Tenant ID field in payload: {tenant_field_in_payload}")
        else:
            print(f"   ⚠️ WARNING: Tenant ID field NOT in payload!")
        
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
                    if tenant_field_verified:
                        verified_value = tenant_field_verified.get("value")
                        if verified_value == tenant_id:
                            print(f"   ✅ Verified: tenant_id was saved correctly: {verified_value}")
                        else:
                            print(f"   ⚠️ WARNING: tenant_id mismatch! Expected: {tenant_id}, Got: {verified_value}")
                    else:
                        print(f"   ⚠️ WARNING: tenant_id field not found in created task!")
                except Exception as verify_exc:
                    print(f"   ⚠️ Could not verify tenant_id after creation: {verify_exc}")
            
            return created_task
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
        
        print(f"🔍 Starting ClickUp delete operation:")
        print(f"   Container ID (list): {container_id}")
        print(f"   Tenant ID to delete: {tenant_id}")
        print(f"   Tenant ID field ID: {tenant_field_id}")
        print(f"   Field map: {field_map}")
        
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
                print(f"📋 Fetched {len(tasks)} tasks from page {page}")
            except Exception as exc:
                print(f"⚠️ Failed to fetch ClickUp tasks: {exc}")
                break
            
            if not tasks:
                print(f"✅ No more tasks to process. Total deleted: {deleted}")
                break
            
            for task in tasks:
                # Get task basic info
                task_id = task.get("id", "")
                task_name = task.get("name", "")
                custom_fields = task.get("custom_fields", [])
                
                # ClickUp list endpoint doesn't return custom field values properly
                # Always fetch individual task details to get actual custom field values
                print(f"🔍 Checking task {task_id} ({task_name}):")
                print(f"   Fetching individual task details to get custom field values...")
                
                try:
                    task_detail_url = f"{self.API_URL}/task/{task_id}"
                    task_detail_response = requests.get(task_detail_url, headers=self._headers(), timeout=20)
                    task_detail_response.raise_for_status()
                    task_detail = task_detail_response.json()
                    custom_fields = task_detail.get("custom_fields", [])
                    print(f"   ✅ Fetched task details, found {len(custom_fields)} custom fields")
                    
                    # Debug: Print the complete task response to see actual structure
                    print(f"   🔍 Complete task response structure:")
                    print(f"   Task ID: {task_detail.get('id', 'N/A')}")
                    print(f"   Task Name: {task_detail.get('name', 'N/A')}")
                    print(f"   Custom Fields Count: {len(custom_fields)}")
                    # Print first custom field as example to see structure
                    if custom_fields:
                        print(f"   First custom field example: {json.dumps(custom_fields[0], indent=2)}")
                except Exception as exc:
                    print(f"   ⚠️ Failed to fetch task details: {exc}")
                    custom_fields = []
                
                print(f"   Looking for field ID: {tenant_field_id}")
                
                # Print all field IDs for debugging
                field_ids_found = [f.get("id", "NO_ID") for f in custom_fields]
                print(f"   Field IDs in task: {field_ids_found}")
                
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
                        
                        # Debug: Print the raw field structure
                        print(f"   ✅ Found tenant_id field!")
                        print(f"   Complete field structure: {json.dumps(field, indent=2)}")
                        print(f"   Raw field value: {field_value} (type: {type(field_value).__name__})")
                        
                        # Handle different value formats
                        # ClickUp short_text fields: value is a string or None
                        if field_value is None:
                            print(f"   ⚠️ WARNING: tenant_id field value is None - task may have been created without tenant_id!")
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
                
                if not tenant_field_found:
                    print(f"   ⚠️ Tenant ID field ({tenant_field_id}) NOT FOUND in task's custom fields!")
                
                # Debug logging
                print(f"   Looking for tenant_id: {tenant_id}")
                print(f"   Found tenant_id in field: {item_tenant_id}")
                print(f"   Match: {item_tenant_id == tenant_id}")
                
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
                else:
                    print(f"⏭️ Skipping task {task_id} - tenant_id mismatch (expected: {tenant_id}, found: {item_tenant_id})")
            
            # Check if more pages
            if len(tasks) < batch_size:
                break
            page += 1
        
        print(f"📊 ClickUp delete operation completed: {deleted} task(s) deleted for tenant {tenant_id}")
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

