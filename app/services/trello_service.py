"""
Trello API Service for Scheduled Calls Integration
"""

import json
import re
from typing import Any, Dict, List, Optional
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
        {"key": "jd_id", "title": "JD ID", "type": "text"},
        {"key": "jd_title", "title": "JD Title", "type": "text"},
        {"key": "jd_summary", "title": "JD Summary", "type": "text"},
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

    def get_board_url(self, board_id: str) -> str:
        """
        Get proper Trello board URL with short ID and board name.
        Fetches board details from Trello API to get shortLink and name.
        """
        try:
            # Fetch board details from Trello API
            url = f"{self.API_URL}/boards/{board_id}"
            params = self._auth_params()
            params.update({
                "fields": "shortLink,shortUrl,name,url"
            })
            
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            board_data = response.json()
            
            # Try to get shortUrl first (most reliable)
            if board_data.get("shortUrl"):
                return board_data["shortUrl"]
            
            # Fallback: Build URL from shortLink and name
            short_link = board_data.get("shortLink", "")
            board_name = board_data.get("name", "")
            
            if short_link:
                # Build proper URL: https://trello.com/b/{shortLink}/{board-name}
                if board_name:
                    # Sanitize board name for URL (lowercase, replace spaces with hyphens)
                    sanitized_name = board_name.lower().replace(" ", "-")
                    sanitized_name = re.sub(r'[^a-z0-9-]', '', sanitized_name)
                    return f"https://trello.com/b/{short_link}/{sanitized_name}"
                else:
                    return f"https://trello.com/b/{short_link}"
            
            # Final fallback: Use stored URL or build from board_id
            return board_data.get("url", self.build_container_url(board_id))
            
        except Exception as e:
            # If API call fails, fallback to basic URL
            return self.build_container_url(board_id)

    def _auth_params(self) -> Dict[str, str]:
        """Get authentication parameters"""
        return {
            "key": self.get_api_key(),
            "token": self.get_api_token(),
        }

    @staticmethod
    def parse_board_id_from_url_or_id(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        v = value.strip()
        if not v:
            return None
        m = re.search(r"(?:https?://)?(?:www\.)?trello\.com/b/([^/?#]+)", v, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return v

    def delete_card(self, card_id: str) -> bool:
        if not card_id:
            return False
        url = f"{self.API_URL}/cards/{card_id}"
        try:
            response = requests.delete(url, params=self._auth_params(), timeout=20)
            if response.status_code in (200, 204):
                return True
            if response.status_code == 404:
                return True
            return False
        except Exception:
            return False

    def delete_board(self, board_id: str) -> bool:
        if not board_id:
            return False
        url = f"{self.API_URL}/boards/{board_id}"
        try:
            response = requests.delete(url, params=self._auth_params(), timeout=30)
            if response.status_code in (200, 204):
                return True
            if response.status_code == 404:
                return True
            return False
        except Exception:
            return False

    def create_container(self, container_name: str, **kwargs) -> Dict[str, str]:
        """
        Create a Trello board for scheduled calls.
        Board is automatically set to public visibility for view-only access.
        """
        url = f"{self.API_URL}/boards"
        params = self._auth_params()
        params.update({
            "name": container_name,
            "defaultLists": "false",
            "prefs_permissionLevel": "public",  # Set board to public
            "prefs_visibility": "public",  # Set visibility to public
        })
        
        try:
            response = requests.post(url, params=params, timeout=20)
            response.raise_for_status()
            board_data = response.json()
            
            # Try multiple ID fields (shortLink is preferred, but id also works)
            board_id = board_data.get("shortLink", "") or board_data.get("id", "")
            if not board_id:
                raise ValueError(f"Trello API did not return board ID. Response: {board_data}")
            
            # After creation, ensure board is public (in case creation params didn't work)
            try:
                update_url = f"{self.API_URL}/boards/{board_id}"
                update_params = self._auth_params()
                update_params.update({
                    "prefs/permissionLevel": "public",
                    "prefs/visibility": "public"
                })
                update_response = requests.put(update_url, params=update_params, timeout=20)
                update_response.raise_for_status()
            except Exception:
                pass
            
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

    def set_board_public(self, board_id: str) -> bool:
        """
        Set Trello board to public visibility.
        Useful for making existing boards public without manual intervention.
        
        Args:
            board_id: Trello board ID (shortLink or long ID)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            url = f"{self.API_URL}/boards/{board_id}"
            params = self._auth_params()
            params.update({
                "prefs/permissionLevel": "public",
                "prefs/visibility": "public"
            })
            
            response = requests.put(url, params=params, timeout=20)
            response.raise_for_status()
            return True
        except Exception as e:
            return False

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
                        field_id = created_field.get("id", "")
                        if field_id:
                            field_map[field_key] = field_id
                except Exception:
                    pass
        
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
                except Exception as e:
                    error_msg = f"Failed to update field {key}: {str(e)}"
                    field_update_errors.append(error_msg)
            
            # If custom fields failed, add data to card description as fallback
            if field_update_errors or not field_map:
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
                except Exception:
                    pass
            
            return card_data
        except Exception as exc:
            return None

    def update_item_jd_context(
        self,
        *,
        item_id: str,
        jd_context: Dict[str, Any],
    ) -> Optional[dict]:
        """
        Persist JD / resume / appointment context to card description (n8n reads for /voice/call/initiate).
        """
        try:
            get_url = f"{self.API_URL}/cards/{item_id}"
            get_params = self._auth_params()
            get_params.update({"fields": "desc"})
            get_resp = requests.get(get_url, params=get_params, timeout=20)
            get_resp.raise_for_status()
            card = get_resp.json() or {}

            existing_desc = str(card.get("desc") or "").strip()
            lines = []
            if jd_context.get("jd_id"):
                lines.append(f"JD ID: {jd_context['jd_id']}")
            if jd_context.get("resume_id"):
                lines.append(f"Resume ID: {jd_context['resume_id']}")
            if jd_context.get("appointment_id"):
                lines.append(f"Appointment ID: {jd_context['appointment_id']}")
            if not lines:
                return {"id": item_id}

            jd_block = "\n".join(lines)
            if existing_desc:
                new_desc = f"{existing_desc}\n{jd_block}"
            else:
                new_desc = jd_block

            update_url = f"{self.API_URL}/cards/{item_id}"
            update_params = self._auth_params()
            update_params.update({"desc": new_desc})
            update_resp = requests.put(update_url, params=update_params, timeout=20)
            update_resp.raise_for_status()
            return update_resp.json()
        except Exception:
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
            return None

    def update_item_call_time_utc(
        self,
        item_id: str,
        call_time_utc: str,
        field_map: Dict[str, str],
    ) -> Optional[dict]:
        """Update scheduled call_time_utc custom field on a card."""
        field_id = field_map.get("call_time_utc")
        if not field_id:
            return None
        url = f"{self.API_URL}/cards/{item_id}/customField/{field_id}/item"
        params = self._auth_params()
        params.update({"value": {"text": call_time_utc}})
        try:
            response = requests.put(url, params=params, timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

    def get_item_call_session_id(
        self,
        *,
        item_id: str,
        container_id: str | None = None,
        field_map: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Resolve call_session_id from a Trello card.

        Lookup order:
        1) Custom field item matching call_session_id field id (if known)
        2) Custom field named "Call Session ID" (if board container_id is provided)
        3) Description line: "Call Session ID: <uuid>"
        """
        field_map = field_map or {}
        known_call_session_field_id = field_map.get("call_session_id")

        card_desc = ""
        try:
            card_url = f"{self.API_URL}/cards/{item_id}"
            card_params = self._auth_params()
            card_params.update({"fields": "desc"})
            card_resp = requests.get(card_url, params=card_params, timeout=20)
            card_resp.raise_for_status()
            card_desc = str((card_resp.json() or {}).get("desc") or "")
        except Exception:
            card_desc = ""

        custom_field_items = []
        try:
            items_url = f"{self.API_URL}/cards/{item_id}/customFieldItems"
            items_params = self._auth_params()
            items_resp = requests.get(items_url, params=items_params, timeout=20)
            items_resp.raise_for_status()
            custom_field_items = items_resp.json() or []
        except Exception:
            custom_field_items = []

        call_session_field_id = known_call_session_field_id
        if not call_session_field_id and container_id:
            try:
                fields_url = f"{self.API_URL}/boards/{container_id}/customFields"
                fields_params = self._auth_params()
                fields_resp = requests.get(fields_url, params=fields_params, timeout=20)
                fields_resp.raise_for_status()
                for field in fields_resp.json() or []:
                    if str(field.get("name") or "").strip().lower() == "call session id":
                        call_session_field_id = field.get("id")
                        break
            except Exception:
                call_session_field_id = None

        if call_session_field_id:
            for item in custom_field_items:
                if item.get("idCustomField") != call_session_field_id:
                    continue
                value = item.get("value") or {}
                text_value = value.get("text") if isinstance(value, dict) else None
                if text_value:
                    return str(text_value).strip()

        if card_desc:
            session_match = re.search(r"Call Session ID:\s*([^\n]+)", card_desc, re.IGNORECASE)
            if session_match:
                parsed = session_match.group(1).strip()
                return parsed or None

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
            return 0
        
        for card in cards:
            card_id = card.get("id", "")
            item_tenant_id = None
            
            # Try to get tenant_id from custom fields first (if available)
            if tenant_field_id:
                custom_fields_url = f"{self.API_URL}/card/{card_id}/customFields"
                custom_fields_params = self._auth_params()
                
                try:
                    custom_fields_response = requests.get(custom_fields_url, params=custom_fields_params, timeout=20)
                    custom_fields_response.raise_for_status()
                    custom_fields = custom_fields_response.json()
                    
                    # Check tenant_id field
                    for field in custom_fields:
                        if field.get("id") == tenant_field_id:
                            field_value = field.get("value", {})
                            if isinstance(field_value, dict):
                                item_tenant_id = field_value.get("text", "").strip()
                            else:
                                item_tenant_id = str(field_value).strip()
                            break
                except Exception:
                    # Continue to description parsing fallback
                    pass
            
            # Fallback: Parse tenant_id from card description if custom fields not available
            if not item_tenant_id:
                card_desc = card.get("desc", "")
                if card_desc:
                    # Use UUID pattern to extract tenant_id from description
                    # Format: "Tenant ID: {uuid}"
                    tenant_pattern = rf"Tenant ID:\s*([0-9a-f]{{8}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{12}})"
                    match = re.search(tenant_pattern, card_desc, re.IGNORECASE)
                    if match:
                        item_tenant_id = match.group(1).strip()
            
            # Delete if tenant_id matches
            if item_tenant_id == tenant_id:
                try:
                    delete_url = f"{self.API_URL}/cards/{card_id}"
                    delete_params = self._auth_params()
                    delete_response = requests.delete(delete_url, params=delete_params, timeout=20)
                    delete_response.raise_for_status()
                    deleted += 1
                except Exception:
                    pass
        
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
        Count pending cards from Trello board that belong to a specific tenant.
        Supports both custom fields (if Power-Ups enabled) and description parsing (fallback).
        """
        tenant_field_id = field_map.get("tenant_id")
        status_field_id = field_map.get("status")
        
        # If custom fields are missing, we'll use description parsing as fallback
        # This is common when Trello Power-Ups are not enabled
        if not tenant_field_id or not status_field_id:
            # Don't raise error - we'll parse from description instead
            pass
        
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
            return 0
        
        for card in cards:
            card_id = card.get("id", "")
            card_desc = card.get("desc", "")
            
            item_tenant_id = None
            item_status = None
            
            # First try custom fields (preferred method) - only if field IDs are available
            if tenant_field_id or status_field_id:
                custom_fields_url = f"{self.API_URL}/card/{card_id}/customFields"
                custom_fields_params = self._auth_params()
                
                try:
                    custom_fields_response = requests.get(custom_fields_url, params=custom_fields_params, timeout=20)
                    custom_fields_response.raise_for_status()
                    custom_fields = custom_fields_response.json()
                    
                    # Check tenant_id and status fields
                    for field in custom_fields:
                        field_id = field.get("id", "")
                        field_value = field.get("value", {})
                        
                        if tenant_field_id and field_id == tenant_field_id:
                            if isinstance(field_value, dict):
                                item_tenant_id = field_value.get("text", "").strip()
                            else:
                                item_tenant_id = str(field_value).strip() if field_value else None
                        elif status_field_id and field_id == status_field_id:
                            if isinstance(field_value, dict):
                                item_status = field_value.get("text", "").strip()
                            else:
                                item_status = str(field_value).strip() if field_value else None
                except Exception:
                    # Continue to description parsing fallback
                    pass
            
            # Fallback to description parsing if custom fields not found or failed
            if not item_tenant_id and card_desc:
                # Parse tenant_id from description
                # Format: "Tenant ID: {uuid}" or "Tenant ID: {value}"
                tenant_pattern = rf"Tenant ID:\s*([^\n]+)"
                tenant_match = re.search(tenant_pattern, card_desc, re.IGNORECASE)
                if tenant_match:
                    item_tenant_id = tenant_match.group(1).strip()
            
            if not item_status and card_desc:
                # Parse status from description
                # Format: "Status: {status}" or "Status: Pending"
                status_pattern = rf"Status:\s*([^\n]+)"
                status_match = re.search(status_pattern, card_desc, re.IGNORECASE)
                if status_match:
                    item_status = status_match.group(1).strip()
            
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
            # Don't raise error - we'll parse from description instead
            pass
        
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
        except Exception as exc:
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
                
                if item_call_session_id and call_session_field_id:
                    formatted_item["column_values"].append({
                        "id": call_session_field_id,
                        "text": item_call_session_id
                    })
                
                # Add status to column_values format
                status_field_id = field_map.get("status")
                
                # Try to find status in custom fields fallback if not found in loop above
                item_status = None
                if status_field_id:
                    # Check if we already found it in loop (we didn't look for it yet)
                    for field in custom_fields:
                        if field.get("id", "") == status_field_id:
                            field_value = field.get("value", {})
                            if isinstance(field_value, dict):
                                item_status = field_value.get("text", "").strip()
                            else:
                                item_status = str(field_value).strip() if field_value else None
                            break
                            
                # Fallback to description
                if not item_status and card_desc:
                    status_match = re.search(r'Status:\s*([^\n]+)', card_desc, re.IGNORECASE)
                    if status_match:
                        item_status = status_match.group(1).strip()
                
                if item_status and status_field_id:
                    formatted_item["column_values"].append({
                        "id": status_field_id,
                        "text": item_status
                    })
                
                items.append(formatted_item)
        
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
                updated = True
            except Exception:
                pass
        
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
            updated = True
            return update_response.json()
        except Exception as exc:
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

    # --- Inbound call log → CRM (tenant boards; separate from scheduled calls) ---

    INBOUND_LIST_NAME_DEFAULT = "Inbound call logs"
    TRELLO_DESC_MAX_CHARS = 16300

    def validate_credentials(self) -> Dict[str, str]:
        url = f"{self.API_URL}/members/me"
        params = self._auth_params()
        params["fields"] = "id,username,fullName"
        response = requests.get(url, params=params, timeout=20)
        if response.status_code == 401:
            raise ValueError("Trello authentication failed — check API key and token.")
        response.raise_for_status()
        data = response.json()
        return {
            "id": data.get("id", ""),
            "username": data.get("username", ""),
            "fullName": data.get("fullName", ""),
        }

    def ensure_inbound_call_logs_list(self, board_id: str, list_name: Optional[str] = None) -> str:
        name = list_name or self.INBOUND_LIST_NAME_DEFAULT
        url = f"{self.API_URL}/boards/{board_id}/lists"
        params = self._auth_params()
        params["filter"] = "open"
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        for lst in response.json():
            if lst.get("name", "").strip() == name:
                return lst["id"]
        create_url = f"{self.API_URL}/lists"
        create_params = self._auth_params()
        create_params.update({"name": name, "idBoard": board_id, "pos": "top"})
        create_resp = requests.post(create_url, params=create_params, timeout=20)
        create_resp.raise_for_status()
        created = create_resp.json()
        lid = created.get("id", "")
        if not lid:
            raise ValueError("Trello did not return a list id after create")
        return lid

    def _truncate_desc(self, text: str) -> str:
        if len(text) <= self.TRELLO_DESC_MAX_CHARS:
            return text
        tail = "\n\n…(truncated)"
        return text[: self.TRELLO_DESC_MAX_CHARS - len(tail)] + tail

    def create_inbound_call_log_card(self, list_id: str, card_name: str, description: str) -> Dict[str, str]:
        url = f"{self.API_URL}/cards"
        params = self._auth_params()
        params.update(
            {
                "idList": list_id,
                "name": card_name[:256] if len(card_name) > 256 else card_name,
                "desc": self._truncate_desc(description or ""),
            }
        )
        response = requests.post(url, params=params, timeout=30)
        if response.status_code == 401:
            raise ValueError("Trello authentication failed while creating card.")
        response.raise_for_status()
        data = response.json()
        cid = data.get("id", "")
        card_url = data.get("shortUrl") or data.get("url") or ""
        if not cid:
            raise ValueError("Trello did not return card id")
        return {"id": cid, "url": card_url}

    def update_inbound_call_log_card(
        self, card_id: str, card_name: Optional[str] = None, description: Optional[str] = None
    ) -> Dict[str, str]:
        url = f"{self.API_URL}/cards/{card_id}"
        params = self._auth_params()
        if card_name is not None:
            params["name"] = card_name[:256] if len(card_name) > 256 else card_name
        if description is not None:
            params["desc"] = self._truncate_desc(description)
        response = requests.put(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return {
            "id": data.get("id", card_id),
            "url": data.get("shortUrl") or data.get("url") or "",
        }

