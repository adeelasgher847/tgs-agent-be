"""
Jira API Service for Scheduled Calls Integration
"""

import json
from typing import Dict, List, Optional
import requests
from app.services.base_crm_service import BaseCRMService
from app.core.security import decrypt_api_key
import base64


class JiraService(BaseCRMService):
    """Service for interacting with Jira API"""

    REQUIRED_FIELDS = [
        {"key": "status", "title": "Status", "type": "select"},
        {"key": "agent_id", "title": "Agent ID", "type": "text"},
        {"key": "call_time_utc", "title": "Call Time UTC", "type": "text"},
        {"key": "tenant_id", "title": "Tenant ID", "type": "text"},
        {"key": "user_id", "title": "User ID", "type": "text"},
        {"key": "batch_id", "title": "Batch ID", "type": "text"},
        {"key": "call_session_id", "title": "Call Session ID", "type": "text"},
        {"key": "phone_number_id", "title": "Phone Number ID", "type": "text"},
        {"key": "email_sent", "title": "Email Sent", "type": "select"},
    ]

    def __init__(self, api_key: str, email: str, server_url: str):
        """
        Initialize Jira service
        
        Args:
            api_key: Jira API token
            email: Jira account email
            server_url: Jira server URL (e.g., https://your-domain.atlassian.net)
        """
        self.api_key = api_key
        self.email = email
        self.server_url = server_url.rstrip("/")

    def get_api_key(self) -> str:
        """Get decrypted API key"""
        return decrypt_api_key(self.api_key) if self.api_key.startswith("eyJ") else self.api_key

    def build_container_url(self, container_id: str) -> str:
        """Build URL for Jira project"""
        return f"{self.server_url}/browse/{container_id}"

    def _headers(self) -> Dict[str, str]:
        """Get API headers with basic auth"""
        api_token = self.get_api_key()
        auth_string = f"{self.email}:{api_token}"
        auth_bytes = auth_string.encode("ascii")
        auth_b64 = base64.b64encode(auth_bytes).decode("ascii")
        
        return {
            "Authorization": f"Basic {auth_b64}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def create_container(self, container_name: str, project_key: Optional[str] = None) -> Dict[str, str]:
        """
        Create a Jira project for scheduled calls.
        Note: Jira projects are typically created via UI, so this may not be fully automated.
        """
        if not project_key:
            raise ValueError("Jira project_key is required")
        
        # Verify project exists
        url = f"{self.server_url}/rest/api/3/project/{project_key}"
        response = requests.get(url, headers=self._headers(), timeout=20)
        response.raise_for_status()
        project_data = response.json()
        
        return {
            "id": project_key,
            "url": self.build_container_url(project_key),
        }

    def ensure_required_fields(self, container_id: str) -> Dict[str, str]:
        """
        Ensure Jira project has required custom fields.
        Note: Custom fields in Jira are typically created via UI.
        This method verifies they exist.
        """
        # Get all custom fields in project
        url = f"{self.server_url}/rest/api/3/field"
        response = requests.get(url, headers=self._headers(), timeout=20)
        response.raise_for_status()
        all_fields = response.json()
        
        # Map fields by name
        field_map = {}
        field_by_name = {f.get("name", "").lower(): f for f in all_fields}
        
        # Check for required fields
        for field_def in self.REQUIRED_FIELDS:
            field_name = field_def["title"]
            field_key = field_def["key"]
            
            # Try to find field by name
            if field_name.lower() in field_by_name:
                field_map[field_key] = field_by_name[field_name.lower()].get("id", "")
            elif field_key == "status":
                # Status is a built-in field
                field_map[field_key] = "status"
        
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
        """Create a scheduled call issue in Jira project"""
        url = f"{self.server_url}/rest/api/3/issue"
        
        # Build fields
        fields = {
            "project": {"key": container_id},
            "summary": f"Scheduled Call: {phone_number}",
            "description": f"Scheduled call for {phone_number} at {call_time_utc}",
            "issuetype": {"name": "Task"},
        }
        
        # Add custom fields
        if "status" in field_map:
            fields["status"] = {"name": "Pending"}
        if "agent_id" in field_map:
            fields[field_map["agent_id"]] = agent_id
        if "call_time_utc" in field_map:
            fields[field_map["call_time_utc"]] = call_time_utc
        if "tenant_id" in field_map:
            fields[field_map["tenant_id"]] = tenant_id
        if "user_id" in field_map:
            fields[field_map["user_id"]] = user_id
        if batch_id and "batch_id" in field_map:
            fields[field_map["batch_id"]] = batch_id
        if phone_number_id and "phone_number_id" in field_map:
            fields[field_map["phone_number_id"]] = phone_number_id
        
        payload = {"fields": fields}
        
        try:
            response = requests.post(url, json=payload, headers=self._headers(), timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            print(f"⚠️ Failed to create Jira issue for {phone_number}: {exc}")
            return None

    def update_item_status(
        self,
        container_id: str,
        item_id: str,
        status: str,
        field_map: Dict[str, str],
    ) -> Optional[dict]:
        """Update issue status in Jira"""
        # Get available transitions
        url = f"{self.server_url}/rest/api/3/issue/{item_id}/transitions"
        response = requests.get(url, headers=self._headers(), timeout=20)
        response.raise_for_status()
        transitions = response.json().get("transitions", [])
        
        # Find transition for status
        transition_id = None
        for transition in transitions:
            if transition.get("to", {}).get("name", "").lower() == status.lower():
                transition_id = transition.get("id")
                break
        
        if not transition_id:
            print(f"⚠️ No transition found for status: {status}")
            return None
        
        # Execute transition
        transition_url = f"{self.server_url}/rest/api/3/issue/{item_id}/transitions"
        payload = {"transition": {"id": transition_id}}
        
        try:
            response = requests.post(transition_url, json=payload, headers=self._headers(), timeout=20)
            response.raise_for_status()
            return {"id": item_id, "status": status}
        except Exception as exc:
            print(f"⚠️ Failed to update Jira issue {item_id} status: {exc}")
            return None

    def update_item_call_session_id(
        self,
        container_id: str,
        item_id: str,
        call_session_id: str,
        field_map: Dict[str, str],
    ) -> Optional[dict]:
        """Update call_session_id field for a Jira issue"""
        session_field_id = field_map.get("call_session_id")
        if not session_field_id:
            return None
        
        url = f"{self.server_url}/rest/api/3/issue/{item_id}"
        payload = {
            "fields": {
                session_field_id: call_session_id
            }
        }
        
        try:
            response = requests.put(url, json=payload, headers=self._headers(), timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            print(f"⚠️ Failed to update call_session_id for Jira issue {item_id}: {exc}")
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
        """Delete issues from Jira project that belong to a specific tenant"""
        tenant_field_id = field_map.get("tenant_id")
        if not tenant_field_id:
            raise ValueError("tenant_id field not found in field map")
        
        deleted = 0
        start_at = 0
        
        while True:
            # Search issues with tenant_id
            url = f"{self.server_url}/rest/api/3/search"
            jql = f"project = {container_id} AND {tenant_field_id} = \"{tenant_id}\""
            payload = {
                "jql": jql,
                "startAt": start_at,
                "maxResults": batch_size,
                "fields": ["id", "key"]
            }
            
            try:
                response = requests.post(url, json=payload, headers=self._headers(), timeout=20)
                response.raise_for_status()
                data = response.json()
                issues = data.get("issues", [])
                total = data.get("total", 0)
            except Exception as exc:
                print(f"⚠️ Failed to search Jira issues: {exc}")
                break
            
            if not issues:
                break
            
            for issue in issues:
                issue_id = issue.get("id", "")
                issue_key = issue.get("key", "")
                
                try:
                    delete_url = f"{self.server_url}/rest/api/3/issue/{issue_id}?deleteSubtasks=true"
                    delete_response = requests.delete(delete_url, headers=self._headers(), timeout=20)
                    delete_response.raise_for_status()
                    deleted += 1
                    print(f"✅ Deleted Jira issue {issue_key} (tenant: {tenant_id})")
                except Exception as exc:
                    print(f"⚠️ Failed to delete Jira issue {issue_key}: {exc}")
            
            # Check if more results
            start_at += len(issues)
            if start_at >= total:
                break
        
        return deleted

    def count_pending_items_for_tenant(
        self,
        container_id: str,
        tenant_id: str,
        field_map: Dict[str, str],
        pending_label: str = "Pending",
        batch_size: int = 100
    ) -> int:
        """Count pending issues from Jira project that belong to a specific tenant"""
        tenant_field_id = field_map.get("tenant_id")
        status_field_id = field_map.get("status")
        if not tenant_field_id or not status_field_id:
            raise ValueError("tenant_id or status field not found in field map")
        
        # Search issues with tenant_id and status
        url = f"{self.server_url}/rest/api/3/search"
        jql = f"project = {container_id} AND {tenant_field_id} = \"{tenant_id}\" AND {status_field_id} = \"{pending_label}\""
        payload = {
            "jql": jql,
            "startAt": 0,
            "maxResults": 0,  # Only get count
            "fields": []
        }
        
        try:
            response = requests.post(url, json=payload, headers=self._headers(), timeout=20)
            response.raise_for_status()
            data = response.json()
            return data.get("total", 0)
        except Exception as exc:
            print(f"⚠️ Failed to count Jira pending issues: {exc}")
            return 0

