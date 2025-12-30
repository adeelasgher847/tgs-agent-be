"""
Jira API Service for Scheduled Calls Integration
"""

import json
import re
import traceback
import hashlib
from typing import Dict, List, Optional, Any
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
    
    @staticmethod
    def normalize_name(name: str) -> str:
        """
        Normalize field name for matching:
        - Strip leading/trailing whitespace
        - Collapse multiple spaces to single
        - Lower-case
        
        Args:
            name: Field name to normalize
            
        Returns:
            Normalized field name
        """
        if not name:
            return ""
        # Collapse multiple whitespace to single space, strip, then lower
        normalized = re.sub(r"\s+", " ", name.strip()).lower()
        return normalized
    
    def build_field_map_from_createmeta(self, container_id: str, issuetype: str = "Task") -> Dict[str, Dict[str, Any]]:
        """
        Build field map from createmeta endpoint (source of truth for project-specific fields).
        
        Args:
            container_id: Jira project key
            issuetype: Issue type name (default: "Task")
            
        Returns:
            Dict mapping normalized_field_name -> {"id": field_id, "name": original_name, "def": field_def}
        """
        createmeta_map = {}
        
        try:
            create_metadata_url = f"{self.server_url}/rest/api/3/issue/createmeta?projectKeys={container_id}&issuetypeNames={issuetype}&expand=projects.issuetypes.fields"
            metadata_response = requests.get(create_metadata_url, headers=self._headers(), timeout=20)
            
            if metadata_response.status_code == 200:
                metadata_data = metadata_response.json()
                if "projects" in metadata_data and len(metadata_data["projects"]) > 0:
                    project_data = metadata_data["projects"][0]
                    if "issuetypes" in project_data and len(project_data["issuetypes"]) > 0:
                        issue_type = project_data["issuetypes"][0]
                        if "fields" in issue_type:
                            for field_id, field_def in issue_type["fields"].items():
                                field_name = field_def.get("name", "")
                                normalized_name = self.normalize_name(field_name)
                                
                                if normalized_name:
                                    # If multiple fields match same normalized name, prefer the one already in map
                                    # (createmeta fields are prioritized)
                                    if normalized_name not in createmeta_map:
                                        createmeta_map[normalized_name] = {
                                            "id": field_id,
                                            "name": field_name,
                                            "def": field_def
                                        }
                                    else:
                                        # Log warning if duplicate found
                                        existing_id = createmeta_map[normalized_name]["id"]
                                        print(f"   ⚠️ Duplicate normalized name '{normalized_name}': {existing_id} vs {field_id}, keeping {existing_id}")
        except Exception as e:
            print(f"   ⚠️ Failed to fetch createmeta fields: {e}")
        
        return createmeta_map

    def _text_to_adf(self, text: str) -> Dict:
        """
        Convert plain text to Atlassian Document Format (ADF) for Jira API v3.
        
        Args:
            text: Plain text string
            
        Returns:
            ADF document structure
        """
        return {
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "paragraph",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ]
        }

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

    def _get_current_user_account_id(self) -> Optional[str]:
        """
        Get current user's account ID (for project lead).
        Uses the email from initialization to get account ID.
        """
        try:
            # Get current user info
            url = f"{self.server_url}/rest/api/3/myself"
            response = requests.get(url, headers=self._headers(), timeout=20)
            response.raise_for_status()
            user_data = response.json()
            account_id = user_data.get("accountId") or user_data.get("accountId")
            if account_id:
                return account_id
            # Fallback: try to get from email
            return None
        except Exception as e:
            print(f"⚠️ Failed to get current user account ID: {e}")
            return None

    def _generate_unique_project_key(self, base_name: str, max_attempts: int = 10) -> str:
        """
        Generate a unique Jira project key from container name.
        Format: SC-{INITIALS} or SC-{NUMBER} if too long.
        Jira rules: 2-10 chars, start with letter, uppercase alphanumeric only.
        
        Args:
            base_name: Container name (e.g., "Scheduled Calls - user@example.com")
            max_attempts: Maximum attempts to find unique key
            
        Returns:
            Unique project key (e.g., "SC-USER1")
        """
        # Extract initials or use hash
        import hashlib
        
        # Try to get meaningful key from email/name
        if "@" in base_name:
            # Extract email part
            email_part = base_name.split("@")[0] if "@" in base_name else base_name
            # Get first few uppercase letters
            initials = "".join([c.upper() for c in email_part if c.isalpha()])[:6]
            if not initials:
                # Fallback: use hash
                hash_val = hashlib.md5(base_name.encode()).hexdigest()[:6].upper()
                initials = "SC" + hash_val[:4]
        else:
            # Extract uppercase letters from name
            initials = "".join([c.upper() for c in base_name if c.isalpha()])[:6]
            if not initials:
                initials = "SC"
        
        # Ensure starts with letter and is uppercase
        if not initials or not initials[0].isalpha():
            if not initials:
                initials = "SC"
            else:
                initials = "SC" + initials
        
        # Limit to 8 chars (leave room for number suffix)
        base_key = initials[:8].upper()
        
        # Ensure base_key is at least 2 chars (Jira minimum)
        if len(base_key) < 2:
            base_key = "SC"
        
        # Try base key first
        for attempt in range(max_attempts):
            if attempt == 0:
                test_key = base_key[:10]  # Jira max is 10 chars
            else:
                # Add number suffix if needed
                suffix = str(attempt)[:2]  # Max 2 digits
                test_key = (base_key[:8] + suffix)[:10]
            
            # Validate format
            if re.match(r'^[A-Z][A-Z0-9]{1,9}$', test_key):
                # Check if key exists
                check_url = f"{self.server_url}/rest/api/3/project/{test_key}"
                try:
                    response = requests.get(check_url, headers=self._headers(), timeout=10)
                    if response.status_code == 404:
                        # Key doesn't exist - we can use it
                        return test_key
                except:
                    # If check fails, assume we can use it
                    return test_key
        
        # Fallback: use hash-based key
        hash_val = hashlib.md5(base_name.encode()).hexdigest()[:8].upper()
        return "SC" + hash_val[:8]

    def _get_available_project_types(self) -> List[Dict]:
        """Get available project types from Jira"""
        try:
            url = f"{self.server_url}/rest/api/3/project/type"
            response = requests.get(url, headers=self._headers(), timeout=20)
            response.raise_for_status()
            project_types = response.json()
            return project_types
        except Exception as e:
            print(f"⚠️ Failed to get project types: {e}")
            # Return default types
            return [
                {"key": "software", "formattedKey": "Software"},
                {"key": "business", "formattedKey": "Business"}
            ]

    def _create_jira_project(self, project_name: str, project_key: str) -> Dict[str, str]:
        """
        Create a new Jira project.
        
        Args:
            project_name: Project name (e.g., "Scheduled Calls - user@example.com")
            project_key: Unique project key (e.g., "SC-USER1")
            
        Returns:
            Dict with project key and URL
            
        Raises:
            ValueError: If project creation fails
        """
        # Get current user account ID for project lead
        account_id = self._get_current_user_account_id()
        
        # Get available project types
        project_types = self._get_available_project_types()
        project_type_key = "software"  # Default to software
        
        # Try to find a valid project type
        for pt in project_types:
            if pt.get("key") in ["software", "business"]:
                project_type_key = pt.get("key")
                break
        
        # Build project creation payload
        payload = {
            "key": project_key,
            "name": project_name,
            "projectTypeKey": project_type_key,
        }
        
        # Add lead account ID if available (required for GDPR strict mode)
        if account_id:
            payload["leadAccountId"] = account_id
        
        url = f"{self.server_url}/rest/api/3/project"
        
        try:
            print(f"🔍 Creating Jira project: {project_name} (key: {project_key})...")
            response = requests.post(url, json=payload, headers=self._headers(), timeout=30)
            
            if response.status_code in [200, 201]:
                project_data = response.json()
                created_key = project_data.get("key", project_key)
                print(f"✅ Successfully created Jira project: {created_key}")
                return {
                    "id": created_key,
                    "url": self.build_container_url(created_key),
                }
            else:
                # Handle errors
                try:
                    error_data = response.json()
                    error_messages = error_data.get("errorMessages", [])
                    errors = error_data.get("errors", {})
                    error_msg = ', '.join(error_messages) if error_messages else str(errors)
                except:
                    error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                
                raise ValueError(f"Failed to create Jira project: {error_msg}")
                
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_messages = error_data.get("errorMessages", [])
                errors = error_data.get("errors", {})
                if error_messages:
                    error_msg += f": {', '.join(error_messages)}"
                if errors:
                    error_msg += f" Errors: {errors}"
            except:
                error_msg += f": {e.response.text[:200]}"
            raise ValueError(f"Failed to create Jira project: {error_msg}")
        except Exception as e:
            raise ValueError(f"Failed to create Jira project: {str(e)}")

    def _create_custom_field(self, field_name: str, field_type: str) -> Optional[str]:
        """
        Create a custom field in Jira.
        
        Args:
            field_name: Name of the field (e.g., "Agent ID")
            field_type: Type of field ("text" or "select")
            
        Returns:
            Field ID if created successfully, None otherwise
        """
        # Map field types to Jira field types
        jira_field_type_map = {
            "text": "com.atlassian.jira.plugin.system.customfieldtypes:textfield",
            "select": "com.atlassian.jira.plugin.system.customfieldtypes:select"
        }
        
        jira_type = jira_field_type_map.get(field_type, jira_field_type_map["text"])
        
        # For select fields, we need to create with options
        if field_type == "select":
            # Create select field with options
            payload = {
                "name": field_name,
                "type": jira_type,
                "searcherKey": "com.atlassian.jira.plugin.system.customfieldtypes:selectsearcher"
            }
        else:
            # Text field
            payload = {
                "name": field_name,
                "type": jira_type,
                "searcherKey": "com.atlassian.jira.plugin.system.customfieldtypes:textsearcher"
            }
        
        url = f"{self.server_url}/rest/api/3/field"
        
        try:
            print(f"🔍 Creating custom field: {field_name} (type: {field_type})...")
            response = requests.post(url, json=payload, headers=self._headers(), timeout=30)
            
            if response.status_code in [200, 201]:
                field_data = response.json()
                field_id = field_data.get("id", "")
                print(f"✅ Successfully created custom field: {field_name} (ID: {field_id})")
                
                # For select fields, add options
                if field_type == "select" and field_id:
                    self._add_select_field_options(field_id, ["Yes", "No"])
                
                return field_id
            else:
                try:
                    error_data = response.json()
                    error_messages = error_data.get("errorMessages", [])
                    errors = error_data.get("errors", {})
                    error_msg = ', '.join(error_messages) if error_messages else str(errors)
                except:
                    error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                
                print(f"⚠️ Failed to create custom field {field_name}: {error_msg}")
                return None
                
        except Exception as e:
            print(f"⚠️ Failed to create custom field {field_name}: {e}")
            return None

    def _add_select_field_options(self, field_id: str, options: List[str]):
        """Add options to a select field"""
        try:
            # Get field configuration
            url = f"{self.server_url}/rest/api/3/field/{field_id}/context"
            response = requests.get(url, headers=self._headers(), timeout=20)
            
            if response.status_code == 200:
                contexts = response.json()
                if contexts:
                    context_id = contexts[0].get("id")
                    # Add options to context
                    options_url = f"{self.server_url}/rest/api/3/field/{field_id}/context/{context_id}/option"
                    for option in options:
                        option_payload = {"value": option}
                        opt_response = requests.post(options_url, json=option_payload, headers=self._headers(), timeout=20)
                        if opt_response.status_code in [200, 201]:
                            print(f"✅ Added option '{option}' to select field")
        except Exception as e:
            print(f"⚠️ Failed to add options to select field: {e}")
            # Non-critical, continue

    def _get_required_fields_for_creation(self, container_id: str) -> Dict[str, Any]:
        """
        Get required fields for creating a Task issue in the project.
        Returns dict of field_id -> default_value
        """
        required_fields = {}
        
        try:
            create_metadata_url = f"{self.server_url}/rest/api/3/issue/createmeta?projectKeys={container_id}&issuetypeNames=Task&expand=projects.issuetypes.fields"
            response = requests.get(create_metadata_url, headers=self._headers(), timeout=20)
            
            if response.status_code == 200:
                metadata_data = response.json()
                if "projects" in metadata_data and len(metadata_data["projects"]) > 0:
                    project_data = metadata_data["projects"][0]
                    if "issuetypes" in project_data and len(project_data["issuetypes"]) > 0:
                        issue_type = project_data["issuetypes"][0]
                        if "fields" in issue_type:
                            for field_id, field_def in issue_type["fields"].items():
                                # Skip standard fields that we set manually (summary, project, issuetype, description)
                                # These should never be in required_fields as we set them explicitly
                                standard_fields = ["summary", "project", "issuetype", "description"]
                                if field_id in standard_fields or field_id.lower() in [f.lower() for f in standard_fields]:
                                    continue
                                
                                # Check if field is required
                                if field_def.get("required", False):
                                    field_name = field_def.get("name", "")
                                    field_schema = field_def.get("schema", {})
                                    field_type = field_schema.get("type", "")
                                    
                                    # Get default value or first allowed value
                                    allowed_values = field_def.get("allowedValues", [])
                                    
                                    if field_type == "option":
                                        # Select field - use first option if available
                                        if allowed_values:
                                            required_fields[field_id] = {"value": allowed_values[0].get("value", allowed_values[0].get("name", ""))}
                                        else:
                                            # Try to get options from field context
                                            try:
                                                context_url = f"{self.server_url}/rest/api/3/field/{field_id}/context"
                                                context_resp = requests.get(context_url, headers=self._headers(), timeout=10)
                                                if context_resp.status_code == 200:
                                                    contexts = context_resp.json()
                                                    if contexts:
                                                        context_id = contexts[0].get("id")
                                                        options_url = f"{self.server_url}/rest/api/3/field/{field_id}/context/{context_id}/option"
                                                        options_resp = requests.get(options_url, headers=self._headers(), timeout=10)
                                                        if options_resp.status_code == 200:
                                                            options = options_resp.json().get("values", [])
                                                            if options:
                                                                required_fields[field_id] = {"value": options[0].get("value", options[0].get("name", ""))}
                                            except:
                                                pass
                                    elif field_type in ["string", "text"]:
                                        # Text field - only set if there's a default value
                                        # Don't set empty string for required text fields - they'll be set in custom fields update step
                                        default_value = field_def.get("defaultValue")
                                        if default_value and default_value != "":
                                            required_fields[field_id] = default_value
                                        # If no default, skip it - we'll handle it in custom fields update step
                                        # This prevents overwriting our manually set fields with empty strings
                                    elif field_type == "number":
                                        required_fields[field_id] = 0
                                    elif field_type == "date":
                                        # Can be omitted or set to current date
                                        pass
                                    
                                    print(f"📋 Found required field: {field_name} ({field_id}) - type: {field_type}")
                                    if field_type == "option" and allowed_values:
                                        print(f"   Available options: {[opt.get('value') or opt.get('name') for opt in allowed_values]}")
        except Exception as e:
            print(f"⚠️ Failed to fetch required fields: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
        
        print(f"📊 Total required fields detected: {len(required_fields)}")
        if required_fields:
            print(f"   Required field IDs: {list(required_fields.keys())}")
        
        return required_fields

    def _get_select_field_value(self, field_id: str, project_key: str, preferred_value: str = "No") -> Optional[str]:
        """
        Get a valid value for a select field.
        Tries to use preferred_value, otherwise returns first available option.
        
        Args:
            field_id: Custom field ID
            project_key: Project key
            preferred_value: Preferred value to use (e.g., "No")
            
        Returns:
            Valid field value or None if field has no options
        """
        try:
            # Get field metadata from issue create metadata (shows available options)
            create_metadata_url = f"{self.server_url}/rest/api/3/issue/createmeta?projectKeys={project_key}&issuetypeNames=Task&expand=projects.issuetypes.fields"
            response = requests.get(create_metadata_url, headers=self._headers(), timeout=20)
            
            if response.status_code == 200:
                metadata_data = response.json()
                if "projects" in metadata_data and len(metadata_data["projects"]) > 0:
                    project_data = metadata_data["projects"][0]
                    if "issuetypes" in project_data and len(project_data["issuetypes"]) > 0:
                        issue_type = project_data["issuetypes"][0]
                        if "fields" in issue_type:
                            fields = issue_type["fields"]
                            if field_id in fields:
                                field_def = fields[field_id]
                                # Get allowed values
                                allowed_values = field_def.get("allowedValues", [])
                                
                                if allowed_values:
                                    # Try to find preferred value (case-insensitive, check both value and name)
                                    for option in allowed_values:
                                        option_value = option.get("value", "")
                                        option_name = option.get("name", "")
                                        
                                        # Check exact match first
                                        if option_value.lower() == preferred_value.lower():
                                            return option_value
                                        if option_name.lower() == preferred_value.lower():
                                            return option_value if option_value else option_name
                                        
                                        # Check partial match (e.g., "No" in "No Email")
                                        if preferred_value.lower() in option_value.lower() or preferred_value.lower() in option_name.lower():
                                            return option_value if option_value else option_name
                                    
                                    # If preferred not found, return first available option
                                    first_option = allowed_values[0].get("value", "")
                                    if first_option:
                                        print(f"⚠️ Preferred value '{preferred_value}' not found in Email Sent field. Available: {[opt.get('value') or opt.get('name') for opt in allowed_values]}. Using '{first_option}' instead.")
                                        return first_option
                                    
                                    # If no value in option, try name
                                    first_name = allowed_values[0].get("name", "")
                                    if first_name:
                                        print(f"⚠️ Using option name '{first_name}' for Email Sent field.")
                                        return first_name
                                
                                # If no allowed values, field might be empty/optional
                                print(f"⚠️ Email Sent field has no allowed values. Field may be optional.")
                                return None
            
            # Fallback: try to get from field context
            try:
                context_url = f"{self.server_url}/rest/api/3/field/{field_id}/context"
                context_response = requests.get(context_url, headers=self._headers(), timeout=20)
                if context_response.status_code == 200:
                    contexts = context_response.json()
                    if contexts:
                        context_id = contexts[0].get("id")
                        options_url = f"{self.server_url}/rest/api/3/field/{field_id}/context/{context_id}/option"
                        options_response = requests.get(options_url, headers=self._headers(), timeout=20)
                        if options_response.status_code == 200:
                            options_data = options_response.json()
                            options = options_data.get("values", [])
                            
                            if options:
                                # Try preferred value
                                for opt in options:
                                    opt_value = opt.get("value", "")
                                    if opt_value.lower() == preferred_value.lower():
                                        return opt_value
                                
                                # Return first option
                                first_opt = options[0].get("value", "")
                                if first_opt:
                                    return first_opt
            except:
                pass
            
            return None
        except Exception as e:
            print(f"⚠️ Failed to get select field options: {e}")
            return None

    def create_container(self, container_name: str, project_key: Optional[str] = None) -> Dict[str, str]:
        """
        Create or verify Jira project exists.
        If project_key is provided, verifies it exists.
        If project_key is not provided, automatically creates a new project with unique key.
        
        Args:
            container_name: Project name (e.g., "Scheduled Calls - user@example.com")
            project_key: Optional - existing Jira project key. If not provided, will auto-create.
            
        Returns:
            Dict with project id (key) and url
            
        Raises:
            ValueError: If project creation/verification fails
        """
        # If project_key provided, verify it exists
        if project_key:
            # Validate project key format (Jira rules: 2-10 chars, start with letter, alphanumeric only)
            if not re.match(r'^[A-Z][A-Z0-9]{1,9}$', project_key):
                raise ValueError(
                    f"Invalid Jira project key format: '{project_key}'. "
                    f"Project keys must be 2-10 characters, start with a letter (A-Z), and contain only uppercase letters and numbers."
                )
            
            # Verify project exists
            check_url = f"{self.server_url}/rest/api/3/project/{project_key}"
            try:
                response = requests.get(check_url, headers=self._headers(), timeout=20)
                
                if response.status_code == 200:
                    # Project exists, return it
                    project_data = response.json()
                    print(f"✅ Verified Jira project exists: {project_key}")
                    return {
                        "id": project_data.get("key", project_key),
                        "url": self.build_container_url(project_data.get("key", project_key)),
                    }
                elif response.status_code == 404:
                    # Project doesn't exist - create it
                    print(f"⚠️ Project '{project_key}' not found. Creating new project...")
                    return self._create_jira_project(container_name, project_key)
                else:
                    # Other error (403, 500, etc.)
                    try:
                        error_data = response.json()
                        error_messages = error_data.get("errorMessages", [])
                        errors = error_data.get("errors", {})
                        error_msg = f"Error checking Jira project: {', '.join(error_messages) if error_messages else str(errors)}"
                    except:
                        error_msg = f"Error checking Jira project: HTTP {response.status_code} - {response.text[:200]}"
                    
                    raise ValueError(
                        f"{error_msg}. "
                        f"Please ensure the project '{project_key}' exists and your API token has proper permissions."
                    )
                    
            except requests.exceptions.RequestException as e:
                raise ValueError(
                    f"Failed to verify Jira project '{project_key}': {str(e)}. "
                    f"Please check your Jira server URL and API credentials."
                )
        else:
            # No project_key provided - check if project with same name exists, otherwise create new
            print(f"🔍 No project_key provided. Checking for existing project with name: {container_name}")
            
            # First, try to find existing project by name
            try:
                # Get all projects and search by name
                projects_url = f"{self.server_url}/rest/api/3/project"
                projects_response = requests.get(projects_url, headers=self._headers(), timeout=20)
                
                if projects_response.status_code == 200:
                    all_projects = projects_response.json()
                    # Search for project with matching name
                    for project in all_projects:
                        if project.get("name", "").strip() == container_name.strip():
                            existing_key = project.get("key", "")
                            print(f"✅ Found existing Jira project with same name: {existing_key}")
                            return {
                                "id": existing_key,
                                "url": self.build_container_url(existing_key),
                            }
                    print(f"   No existing project found with name '{container_name}', creating new one...")
            except Exception as search_error:
                print(f"   ⚠️ Could not search for existing projects: {search_error}, proceeding with creation...")
            
            # No existing project found - create new one
            generated_key = self._generate_unique_project_key(container_name)
            try:
                return self._create_jira_project(container_name, generated_key)
            except ValueError as create_error:
                # If creation fails due to name conflict, try to find the project again
                error_str = str(create_error)
                if "project with that name already exists" in error_str.lower() or "projectname" in error_str.lower():
                    print(f"   ⚠️ Project name conflict detected. Searching for existing project...")
                    try:
                        projects_url = f"{self.server_url}/rest/api/3/project"
                        projects_response = requests.get(projects_url, headers=self._headers(), timeout=20)
                        if projects_response.status_code == 200:
                            all_projects = projects_response.json()
                            for project in all_projects:
                                if project.get("name", "").strip() == container_name.strip():
                                    existing_key = project.get("key", "")
                                    print(f"✅ Found existing Jira project: {existing_key}")
                                    return {
                                        "id": existing_key,
                                        "url": self.build_container_url(existing_key),
                                    }
                    except Exception as retry_error:
                        print(f"   ⚠️ Failed to find existing project: {retry_error}")
                
                # Re-raise the original error if we couldn't find existing project
                raise create_error

    def ensure_required_fields(self, container_id: str) -> Dict[str, str]:
        """
        Get existing custom field IDs. If fields don't exist, automatically create them.
        Uses createmeta as source of truth, with robust name normalization to prevent duplicates.
        
        Args:
            container_id: Jira project key
            
        Returns:
            Dict mapping field keys to field IDs
        """
        field_map = {}
        missing_fields = []
        
        try:
            # Step 1: Get createmeta fields (source of truth - fields available for Task in this project)
            print(f"🔍 Fetching createmeta fields for project {container_id}...")
            createmeta_map = self.build_field_map_from_createmeta(container_id, "Task")
            print(f"   ✅ Found {len(createmeta_map)} fields in createmeta")
            
            # Step 2: Get all global fields as fallback
            print(f"🔍 Fetching global fields as fallback...")
            url = f"{self.server_url}/rest/api/3/field"
            response = requests.get(url, headers=self._headers(), timeout=20)
            response.raise_for_status()
            all_fields = response.json()
            
            # Build normalized map of global fields (for fallback matching)
            global_field_by_normalized_name = {}
            for field in all_fields:
                field_name = field.get("name", "")
                normalized_name = self.normalize_name(field_name)
                if normalized_name:
                    # If duplicate, prefer the one already in map (first one wins)
                    if normalized_name not in global_field_by_normalized_name:
                        global_field_by_normalized_name[normalized_name] = field
            
            print(f"   ✅ Found {len(global_field_by_normalized_name)} unique global fields (normalized)")
            
            # Step 3: Map required fields, prioritizing createmeta
            for field_def in self.REQUIRED_FIELDS:
                field_name = field_def["title"]
                field_key = field_def["key"]
                field_type = field_def["type"]
                normalized_field_name = self.normalize_name(field_name)
                
                # Status is a built-in field
                if field_key == "status":
                    field_map[field_key] = "status"
                    print(f"   ✅ Mapped {field_key} -> status (built-in)")
                    continue
                
                matched_field_id = None
                matched_source = None
                
                # Priority 1: Check createmeta first (source of truth)
                if normalized_field_name in createmeta_map:
                    matched_field_id = createmeta_map[normalized_field_name]["id"]
                    matched_source = "createmeta"
                    original_name = createmeta_map[normalized_field_name]["name"]
                    print(f"   ✅ Matched '{field_name}' -> {matched_field_id} (createmeta, original: '{original_name}')")
                
                # Priority 2: Fallback to global fields
                elif normalized_field_name in global_field_by_normalized_name:
                    matched_field = global_field_by_normalized_name[normalized_field_name]
                    matched_field_id = matched_field.get("id", "")
                    matched_source = "global"
                    original_name = matched_field.get("name", "")
                    print(f"   ✅ Matched '{field_name}' -> {matched_field_id} (global, original: '{original_name}')")
                
                # Priority 3: Check if multiple fields match (safety check)
                if not matched_field_id:
                    # Check for partial matches in createmeta
                    createmeta_matches = [k for k in createmeta_map.keys() if normalized_field_name in k or k in normalized_field_name]
                    if createmeta_matches:
                        # Use the first match from createmeta
                        matched_normalized = createmeta_matches[0]
                        matched_field_id = createmeta_map[matched_normalized]["id"]
                        matched_source = "createmeta (partial)"
                        original_name = createmeta_map[matched_normalized]["name"]
                        print(f"   ⚠️ Partial match '{field_name}' -> {matched_field_id} (createmeta partial, original: '{original_name}')")
                
                if matched_field_id:
                    field_map[field_key] = matched_field_id
                else:
                    # Field not found in either source - mark for creation
                    missing_fields.append({"name": field_name, "type": field_type, "key": field_key})
                    print(f"   ⚠️ Field '{field_name}' not found, will create new")
            
            # Step 4: Create missing fields only if not found in BOTH sources
            if missing_fields:
                print(f"⚠️ {len(missing_fields)} custom fields missing. Creating them automatically...")
                for field_info in missing_fields:
                    # Double-check: maybe field was created between checks
                    normalized_name = self.normalize_name(field_info["name"])
                    
                    # Re-check createmeta after potential creation
                    if normalized_name in createmeta_map:
                        field_id = createmeta_map[normalized_name]["id"]
                        field_map[field_info["key"]] = field_id
                        print(f"   ✅ Found existing field '{field_info['name']}' -> {field_id} (after re-check)")
                        continue
                    
                    # Create new field
                    field_id = self._create_custom_field(field_info["name"], field_info["type"])
                    if field_id:
                        field_map[field_info["key"]] = field_id
                        print(f"   ✅ Created and mapped field: {field_info['name']} -> {field_id}")
                    else:
                        print(f"   ❌ Failed to create field: {field_info['name']}")
                        # Still raise error if critical field creation fails
                        raise ValueError(f"Failed to create required custom field: {field_info['name']}")
            
            print(f"📊 Jira field_map: {field_map} (Total fields: {len(field_map)})")
            return field_map
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"Failed to fetch Jira fields: HTTP {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_messages = error_data.get("errorMessages", [])
                if error_messages:
                    error_msg += f" - {', '.join(error_messages)}"
            except:
                error_msg += f" - {e.response.text[:200]}"
            
            raise ValueError(f"{error_msg}. Please check your Jira API credentials and permissions.")
            
        except requests.exceptions.RequestException as e:
            raise ValueError(
                f"Network error fetching Jira fields: {str(e)}. "
                f"Please check your Jira server URL and network connection."
            )
            
        except ValueError:
            # Re-raise ValueError (missing fields or creation failures)
            raise
            
        except Exception as e:
            raise ValueError(
                f"Unexpected error fetching Jira fields: {str(e)}. "
                f"Please ensure your Jira instance is accessible and API credentials are correct."
            )

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
        Create a scheduled call issue in Jira project.
        Uses dynamic field_map (from ensure_required_fields) instead of hardcoded IDs.
        Does NOT set status during creation - will use transition API after creation.
        """
        url = f"{self.server_url}/rest/api/3/issue"
        
        # Validate field_map has all required fields (email_sent is optional)
        required_field_keys = ["agent_id", "call_time_utc", "tenant_id", "user_id"]
        missing_in_map = [key for key in required_field_keys if key not in field_map]
        if missing_in_map:
            raise ValueError(f"Missing required fields in field_map: {', '.join(missing_in_map)}")
        
        # Step 1: Get required fields for creation (fields that MUST be set during creation)
        required_fields = self._get_required_fields_for_creation(container_id)
        
        # Step 2: Create issue with basic fields + required fields
        # Build description with all fields including status (similar to Trello/ClickUp)
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
        
        description_text = "\n".join(desc_lines)
        
        basic_fields = {
            "project": {"key": container_id},
            "summary": f"Scheduled Call: {phone_number}",
            "description": self._text_to_adf(description_text),  # Convert to ADF format with all fields
            "issuetype": {"name": "Task"},
        }
        
        # Add required fields to basic_fields (these MUST be set during creation)
        # Map required field IDs to our field values by matching field names
        try:
            create_metadata_url = f"{self.server_url}/rest/api/3/issue/createmeta?projectKeys={container_id}&issuetypeNames=Task&expand=projects.issuetypes.fields"
            metadata_resp = requests.get(create_metadata_url, headers=self._headers(), timeout=10)
            if metadata_resp.status_code == 200:
                metadata = metadata_resp.json()
                if "projects" in metadata and len(metadata["projects"]) > 0:
                    project = metadata["projects"][0]
                    if "issuetypes" in project and len(project["issuetypes"]) > 0:
                        issue_type = project["issuetypes"][0]
                        if "fields" in issue_type:
                            for req_field_id in required_fields.keys():
                                if req_field_id in issue_type["fields"]:
                                    field_def = issue_type["fields"][req_field_id]
                                    field_name = field_def.get("name", "").strip().lower()
                                    
                                    print(f"   🔍 Processing required field: '{field_name}' ({req_field_id})")
                                    
                                    # Check if this is Email Sent field
                                    if "email" in field_name and "sent" in field_name:
                                        # This is Email Sent - set to "No"
                                        email_sent_value = self._get_select_field_value(req_field_id, container_id, preferred_value="No")
                                        if email_sent_value:
                                            basic_fields[req_field_id] = {"value": email_sent_value}
                                            print(f"   ✅ Set required Email Sent field {req_field_id} to '{email_sent_value}'")
                                        else:
                                            # If no value found, use first available option
                                            allowed_values = field_def.get("allowedValues", [])
                                            if allowed_values:
                                                first_option = allowed_values[0].get("value") or allowed_values[0].get("name", "")
                                                basic_fields[req_field_id] = {"value": first_option}
                                                print(f"   ⚠️ Using first available option for Email Sent: '{first_option}'")
                                    # Check if this is Impact field (or other required select fields)
                                    elif field_def.get("schema", {}).get("type") == "option":
                                        # Select field - use first available option
                                        allowed_values = field_def.get("allowedValues", [])
                                        if allowed_values:
                                            first_option = allowed_values[0].get("value") or allowed_values[0].get("name", "")
                                            basic_fields[req_field_id] = {"value": first_option}
                                            print(f"   ✅ Set required select field '{field_name}' ({req_field_id}) to '{first_option}'")
                                    # For other required fields, map to our actual values by matching field names
                                    elif req_field_id not in basic_fields:
                                        # Try to match required field to our field_map by name
                                        matched = False
                                        
                                        # Match by field name patterns
                                        if "agent" in field_name and "id" in field_name:
                                            basic_fields[req_field_id] = agent_id
                                            print(f"   ✅ Mapped required field '{field_name}' ({req_field_id}) → agent_id = {agent_id}")
                                            matched = True
                                        elif "call" in field_name and "time" in field_name and "utc" in field_name:
                                            basic_fields[req_field_id] = call_time_utc
                                            print(f"   ✅ Mapped required field '{field_name}' ({req_field_id}) → call_time_utc = {call_time_utc}")
                                            matched = True
                                        elif "tenant" in field_name and "id" in field_name:
                                            basic_fields[req_field_id] = tenant_id
                                            print(f"   ✅ Mapped required field '{field_name}' ({req_field_id}) → tenant_id = {tenant_id}")
                                            matched = True
                                        elif "user" in field_name and "id" in field_name and "agent" not in field_name and "tenant" not in field_name:
                                            basic_fields[req_field_id] = user_id
                                            print(f"   ✅ Mapped required field '{field_name}' ({req_field_id}) → user_id = {user_id}")
                                            matched = True
                                        elif "batch" in field_name and "id" in field_name and batch_id:
                                            basic_fields[req_field_id] = batch_id
                                            print(f"   ✅ Mapped required field '{field_name}' ({req_field_id}) → batch_id = {batch_id}")
                                            matched = True
                                        elif "phone" in field_name and "number" in field_name and "id" in field_name and phone_number_id:
                                            basic_fields[req_field_id] = phone_number_id
                                            print(f"   ✅ Mapped required field '{field_name}' ({req_field_id}) → phone_number_id = {phone_number_id}")
                                            matched = True
                                        elif "call" in field_name and "session" in field_name and "id" in field_name:
                                            basic_fields[req_field_id] = ""  # Leave blank initially
                                            print(f"   ✅ Mapped required field '{field_name}' ({req_field_id}) → call_session_id (blank)")
                                            matched = True
                                        
                                        # If not matched, use default value
                                        if not matched:
                                            default_value = required_fields[req_field_id]
                                            basic_fields[req_field_id] = default_value
                                            print(f"   ⚠️ No match found for '{field_name}' ({req_field_id}), using default value: {default_value}")
        except Exception as e:
            print(f"   ⚠️ Failed to map required fields: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            # Fallback: add required fields with default values
            for field_id, field_value in required_fields.items():
                if field_id not in basic_fields:
                    basic_fields[field_id] = field_value
                    print(f"   ✅ Added required field {field_id} to creation payload (fallback)")
        
        # Step 3: Prepare ALL custom fields for update step (fields that are NOT required/on create screen)
        # Get list of required field IDs to exclude from update
        required_field_ids = set(required_fields.keys())
        custom_fields_to_update = {}
        
        # Set all fields from field_map (except status and required fields which are handled during creation)
        if "agent_id" in field_map and field_map["agent_id"]:
            # Only add if not in required fields (not set during creation)
            if field_map["agent_id"] not in required_field_ids:
                custom_fields_to_update[field_map["agent_id"]] = agent_id
                print(f"   📦 Added agent_id field: {field_map['agent_id']} = {agent_id}")
            else:
                print(f"   ⏭️ Skipped agent_id (already set in required fields: {field_map['agent_id']})")
        
        if "call_time_utc" in field_map and field_map["call_time_utc"]:
            if field_map["call_time_utc"] not in required_field_ids:
                custom_fields_to_update[field_map["call_time_utc"]] = call_time_utc
                print(f"   📦 Added call_time_utc field: {field_map['call_time_utc']} = {call_time_utc}")
            else:
                print(f"   ⏭️ Skipped call_time_utc (already set in required fields: {field_map['call_time_utc']})")
        
        if "tenant_id" in field_map and field_map["tenant_id"]:
            if field_map["tenant_id"] not in required_field_ids:
                custom_fields_to_update[field_map["tenant_id"]] = tenant_id
                print(f"   📦 Added tenant_id field: {field_map['tenant_id']} = {tenant_id}")
            else:
                print(f"   ⏭️ Skipped tenant_id (already set in required fields: {field_map['tenant_id']})")
        
        if "user_id" in field_map and field_map["user_id"]:
            if field_map["user_id"] not in required_field_ids:
                custom_fields_to_update[field_map["user_id"]] = user_id
                print(f"   📦 Added user_id field: {field_map['user_id']} = {user_id}")
            else:
                print(f"   ⏭️ Skipped user_id (already set in required fields: {field_map['user_id']})")
        
        if batch_id and "batch_id" in field_map and field_map["batch_id"]:
            if field_map["batch_id"] not in required_field_ids:
                custom_fields_to_update[field_map["batch_id"]] = batch_id
                print(f"   📦 Added batch_id field: {field_map['batch_id']} = {batch_id}")
            else:
                print(f"   ⏭️ Skipped batch_id (already set in required fields: {field_map['batch_id']})")
        
        if phone_number_id and "phone_number_id" in field_map and field_map["phone_number_id"]:
            if field_map["phone_number_id"] not in required_field_ids:
                custom_fields_to_update[field_map["phone_number_id"]] = phone_number_id
                print(f"   📦 Added phone_number_id field: {field_map['phone_number_id']} = {phone_number_id}")
            else:
                print(f"   ⏭️ Skipped phone_number_id (already set in required fields: {field_map['phone_number_id']})")
        
        if "call_session_id" in field_map and field_map["call_session_id"]:
            # Only add if not already in basic_fields (not required)
            if field_map["call_session_id"] not in required_field_ids and field_map["call_session_id"] not in basic_fields:
                custom_fields_to_update[field_map["call_session_id"]] = ""  # Leave blank initially
                print(f"   📦 Added call_session_id field: {field_map['call_session_id']} = (blank)")
            else:
                print(f"   ⏭️ Skipped call_session_id (already set in required fields)")
        
        # Email Sent: Only add to update if it's NOT in required fields (already set in basic_fields)
        if "email_sent" in field_map and field_map["email_sent"]:
            email_sent_field_id = field_map["email_sent"]
            # Check if this field is already in basic_fields (required field) OR in required_field_ids
            if email_sent_field_id not in basic_fields and email_sent_field_id not in required_field_ids:
                # Not required, add to update step
                email_sent_value = self._get_select_field_value(email_sent_field_id, container_id, preferred_value="No")
                if email_sent_value:
                    custom_fields_to_update[email_sent_field_id] = {"value": email_sent_value}
                    print(f"   📦 Added email_sent field to update: {email_sent_field_id} = {email_sent_value}")
                else:
                    print(f"   ⚠️ Could not set Email Sent field - no valid value found")
            else:
                print(f"   ✅ Email Sent field already set in required fields (skipping update step)")
        
        basic_payload = {"fields": basic_fields}
        
        try:
            print(f"🔍 Creating Jira issue for {phone_number} in project {container_id}...")
            print(f"   URL: {url}")
            print(f"   Step 1 - Basic payload: {json.dumps(basic_payload, indent=2)}")
            
            response = requests.post(url, json=basic_payload, headers=self._headers(), timeout=20)
            
            if response.status_code in [200, 201]:
                issue_data = response.json()
                issue_key = issue_data.get("key", "")
                issue_id = issue_data.get("id", "")
                print(f"✅ Successfully created Jira issue {issue_key} (ID: {issue_id}) for {phone_number}")
                
                # Step 2: Update issue with custom fields (fields not on create screen)
                if custom_fields_to_update:
                    update_url = f"{self.server_url}/rest/api/3/issue/{issue_id}"
                    update_payload = {"fields": custom_fields_to_update}
                    print(f"   Step 2 - Updating custom fields: {json.dumps(update_payload, indent=2)}")
                    print(f"   Total fields to update: {len(custom_fields_to_update)}")
                    
                    try:
                        update_response = requests.put(update_url, json=update_payload, headers=self._headers(), timeout=20)
                        print(f"   Update response status: {update_response.status_code}")
                        
                        if update_response.status_code in [200, 204]:
                            print(f"✅ Successfully updated custom fields for issue {issue_key}")
                        else:
                            # Log error but don't fail - issue was created successfully
                            try:
                                update_error = update_response.json()
                                error_messages = update_error.get("errorMessages", [])
                                errors_dict = update_error.get("errors", {})
                                print(f"⚠️ Failed to update custom fields for issue {issue_key}:")
                                print(f"   Error Messages: {error_messages}")
                                print(f"   Errors: {errors_dict}")
                                print(f"   Full response: {update_response.text[:500]}")
                                
                                # If update fails due to screen issue, try to add fields one by one
                                print(f"   🔄 Attempting to update fields individually...")
                                successful_updates = 0
                                failed_updates = 0
                                
                                for field_id, field_value in custom_fields_to_update.items():
                                    try:
                                        single_field_payload = {"fields": {field_id: field_value}}
                                        print(f"   🔍 Updating field {field_id} with value: {field_value}")
                                        single_update = requests.put(update_url, json=single_field_payload, headers=self._headers(), timeout=20)
                                        if single_update.status_code in [200, 204]:
                                            print(f"   ✅ Successfully updated field {field_id}")
                                            successful_updates += 1
                                        else:
                                            try:
                                                single_error = single_update.json()
                                                print(f"   ⚠️ Failed to update field {field_id}: {single_error}")
                                            except:
                                                print(f"   ⚠️ Failed to update field {field_id}: HTTP {single_update.status_code} - {single_update.text[:200]}")
                                            failed_updates += 1
                                    except Exception as single_field_error:
                                        print(f"   ❌ Error updating field {field_id}: {single_field_error}")
                                        failed_updates += 1
                                
                                print(f"   📊 Update summary: {successful_updates} successful, {failed_updates} failed")
                            except Exception as parse_error:
                                print(f"⚠️ Failed to parse update error for issue {issue_key}: {parse_error}")
                                print(f"   Response status: {update_response.status_code}")
                                print(f"   Response text: {update_response.text[:500]}")
                    except Exception as update_error:
                        print(f"⚠️ Error updating custom fields for issue {issue_key}: {update_error}")
                        import traceback
                        print(f"   Traceback: {traceback.format_exc()}")
                        # Don't fail - issue was created successfully
                else:
                    print(f"⚠️ No custom fields to update for issue {issue_key}")
                
                # Set status using transition API - try "Pending" first, then fallback to available status
                try:
                    # Try "Pending" first
                    status_result = self.update_item_status(
                        container_id=container_id,
                        item_id=issue_id,
                        status="Pending",
                        field_map={}  # Not needed for transitions
                    )
                    if status_result:
                        print(f"✅ Set status to 'Pending' for issue {issue_key}")
                    else:
                        # If "Pending" not available, try to get first available status (usually "To Do" or "In Progress")
                        print(f"⚠️ 'Pending' status not available. Trying to get available statuses...")
                        transitions_url = f"{self.server_url}/rest/api/3/issue/{issue_id}/transitions"
                        transitions_response = requests.get(transitions_url, headers=self._headers(), timeout=20)
                        if transitions_response.status_code == 200:
                            transitions = transitions_response.json().get("transitions", [])
                            if transitions:
                                # Use first available transition (usually "To Do" or initial status)
                                first_transition = transitions[0]
                                transition_id = first_transition.get("id")
                                target_status = first_transition.get("to", {}).get("name", "Unknown")
                                
                                # Execute transition
                                transition_execute_url = f"{self.server_url}/rest/api/3/issue/{issue_id}/transitions"
                                transition_payload = {"transition": {"id": transition_id}}
                                transition_exec_response = requests.post(transition_execute_url, json=transition_payload, headers=self._headers(), timeout=20)
                                if transition_exec_response.status_code in [200, 204]:
                                    print(f"✅ Set status to '{target_status}' for issue {issue_key} (Pending not available)")
                                else:
                                    print(f"⚠️ Could not set status for issue {issue_key}")
                            else:
                                print(f"⚠️ No transitions available for issue {issue_key}")
                        else:
                            print(f"⚠️ Could not fetch transitions for issue {issue_key}")
                except Exception as status_error:
                    print(f"⚠️ Error setting status for issue {issue_key}: {status_error}")
                    # Don't fail the entire operation if status transition fails
                
                return issue_data
            else:
                # HTTP error - detailed logging
                try:
                    error_data = response.json()
                    error_messages = error_data.get("errorMessages", [])
                    errors_dict = error_data.get("errors", {})
                except:
                    error_messages = []
                    errors_dict = {}
                
                error_detail = {
                    "operation": "create_issue",
                    "phone_number": phone_number,
                    "container_id": container_id,
                    "url": url,
                    "status_code": response.status_code,
                    "response": response.text[:500],
                    "error_messages": error_messages,
                    "errors": errors_dict,
                    "payload": basic_payload
                }
                
                print(f"❌ Failed to create Jira issue for {phone_number}:")
                print(f"   Status: {response.status_code}")
                print(f"   Error Messages: {error_messages}")
                print(f"   Errors: {errors_dict}")
                print(f"   Response: {response.text[:500]}")
                print(f"   Payload: {json.dumps(basic_payload, indent=2)}")
                
                return None
                
        except requests.exceptions.HTTPError as e:
            error_detail = {
                "operation": "create_issue",
                "phone_number": phone_number,
                "container_id": container_id,
                "url": url,
                "status_code": e.response.status_code if e.response else None,
                "response": e.response.text[:500] if e.response else str(e),
                "error": str(e),
                "payload": basic_payload
            }
            
            print(f"❌ HTTP error creating Jira issue for {phone_number}:")
            print(f"   Status: {error_detail['status_code']}")
            print(f"   Response: {error_detail['response']}")
            print(f"   Error: {error_detail['error']}")
            if e.response:
                try:
                    error_data = e.response.json()
                    print(f"   Error Messages: {error_data.get('errorMessages', [])}")
                    print(f"   Errors: {error_data.get('errors', {})}")
                except:
                    pass
            
            return None
            
        except requests.exceptions.RequestException as e:
            error_detail = {
                "operation": "create_issue",
                "phone_number": phone_number,
                "container_id": container_id,
                "url": url,
                "error_type": type(e).__name__,
                "error": str(e),
                "payload": basic_payload
            }
            
            print(f"❌ Network error creating Jira issue for {phone_number}:")
            print(f"   Error Type: {error_detail['error_type']}")
            print(f"   Error: {error_detail['error']}")
            
            return None
            
        except Exception as exc:
            error_detail = {
                "operation": "create_issue",
                "phone_number": phone_number,
                "container_id": container_id,
                "url": url,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "payload": basic_payload
            }
            
            print(f"❌ Unexpected error creating Jira issue for {phone_number}:")
            print(f"   Error Type: {error_detail['error_type']}")
            print(f"   Error: {error_detail['error']}")
            print(f"   Traceback: {traceback.format_exc()}")
            
            return None

    def update_item_status(
        self,
        container_id: str,
        item_id: str,
        status: str,
        field_map: Dict[str, str],
    ) -> Optional[dict]:
        """
        Update issue status in Jira using transition API.
        Fetches available transitions, finds the one that moves to target status, and executes it.
        """
        # Get available transitions
        transitions_url = f"{self.server_url}/rest/api/3/issue/{item_id}/transitions"
        try:
            response = requests.get(transitions_url, headers=self._headers(), timeout=20)
            response.raise_for_status()
            transitions = response.json().get("transitions", [])
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_msg += f": {', '.join(error_data.get('errorMessages', []))}"
            except:
                error_msg += f": {e.response.text[:200]}"
            print(f"⚠️ Failed to fetch transitions for issue {item_id}: {error_msg}")
            return None
        except Exception as e:
            print(f"⚠️ Failed to fetch transitions for issue {item_id}: {e}")
            return None
        
        # Find transition for status (case-insensitive match)
        transition_id = None
        for transition in transitions:
            target_status = transition.get("to", {}).get("name", "")
            if target_status.lower() == status.lower():
                transition_id = transition.get("id")
                break
        
        if not transition_id:
            available_statuses = [t.get("to", {}).get("name", "") for t in transitions]
            print(f"⚠️ No transition found for status '{status}'. Available: {', '.join(available_statuses)}")
            return None
        
        # Execute transition
        transition_execute_url = f"{self.server_url}/rest/api/3/issue/{item_id}/transitions"
        payload = {"transition": {"id": transition_id}}
        
        try:
            response = requests.post(transition_execute_url, json=payload, headers=self._headers(), timeout=20)
            response.raise_for_status()
            return {"id": item_id, "status": status}
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_msg += f": {', '.join(error_data.get('errorMessages', []))}"
            except:
                error_msg += f": {e.response.text[:200]}"
            print(f"⚠️ Failed to execute transition for issue {item_id}: {error_msg}")
            return None
        except Exception as e:
            print(f"⚠️ Failed to update Jira issue {item_id} status: {e}")
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
        
        deleted = 0
        
        # Try JQL query first if custom field is available
        if tenant_field_id:
            try:
                deleted = self._delete_by_jql(container_id, tenant_id, tenant_field_id, batch_size)
                if deleted > 0:
                    return deleted
            except Exception as exc:
                print(f"⚠️ JQL query failed, falling back to description parsing: {exc}")
        
        # Fallback: Fetch all issues and parse descriptions
        deleted = self._delete_by_description_parsing(container_id, tenant_id, batch_size)
        
        return deleted
    
    def _delete_by_jql(
        self,
        container_id: str,
        tenant_id: str,
        tenant_field_id: str,
        batch_size: int = 50
    ) -> int:
        """Delete issues using JQL query with custom field"""
        deleted = 0
        start_at = 0
        
        while True:
            # Search issues with tenant_id - Use /rest/api/3/search/jql
            url = f"{self.server_url}/rest/api/3/search/jql"
            jql = f"project = {container_id} AND {tenant_field_id} = \"{tenant_id}\""
            # For /search/jql endpoint: body has jql field, other params in query string
            payload = {"jql": jql}
            params = {
                "startAt": start_at,
                "maxResults": batch_size,
                "fields": "id,key"
            }
            
            try:
                response = requests.post(
                    url, 
                    json=payload,  # JSON body with jql field
                    params=params,  # Other params in query string
                    headers=self._headers(),
                    timeout=20
                )
                response.raise_for_status()
                data = response.json()
                issues = data.get("issues", [])
                total = data.get("total", 0)
            except Exception as exc:
                print(f"⚠️ Failed to search Jira issues: {exc}")
                raise
            
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
                except Exception as exc:
                    print(f"⚠️ Failed to delete Jira issue {issue_key}: {exc}")
            
            # Check if more results
            start_at += len(issues)
            if start_at >= total:
                break
        
        return deleted
    
    def _delete_by_description_parsing(
        self,
        container_id: str,
        tenant_id: str,
        batch_size: int = 50
    ) -> int:
        """Delete issues by parsing description (fallback when custom fields not available)"""
        deleted = 0
        start_at = 0
        
        while True:
            # Fetch all issues from project
            # Use /rest/api/3/search/jql endpoint (410 Gone error for /rest/api/3/search)
            url = f"{self.server_url}/rest/api/3/search/jql"
            jql = f"project = {container_id}"
            # For /rest/api/3/search/jql: fields parameter goes in body as array
            payload = {
                "jql": jql,
                "fields": ["id", "key", "description"]  # fields as array in body
            }
            params = {
                "startAt": start_at,
                "maxResults": batch_size
            }
            
            try:
                response = requests.post(
                    url,
                    json=payload,
                    params=params,
                    headers=self._headers(),
                    timeout=20
                )
                response.raise_for_status()
                data = response.json()
                issues = data.get("issues", [])
                total = data.get("total", 0)
            except Exception as exc:
                print(f"⚠️ Failed to fetch Jira issues: {exc}")
                break
            
            if not issues:
                break
            
            # Check each issue's description for matching tenant_id
            for issue in issues:
                issue_id = issue.get("id", "")
                issue_key = issue.get("key", "")
                fields = issue.get("fields", {})
                description = fields.get("description")
                
                if not description:
                    continue
                
                # Convert ADF to text if needed
                description_text = self._adf_to_text(description) if isinstance(description, dict) else str(description)
                
                # Check if tenant_id matches in description (using UUID pattern)
                tenant_pattern = rf"Tenant ID:\s*({tenant_id})"
                match = re.search(tenant_pattern, description_text, re.IGNORECASE)
                if match:
                    # Delete this issue
                    try:
                        delete_url = f"{self.server_url}/rest/api/3/issue/{issue_id}?deleteSubtasks=true"
                        delete_response = requests.delete(delete_url, headers=self._headers(), timeout=20)
                        delete_response.raise_for_status()
                        deleted += 1
                    except requests.exceptions.HTTPError as exc:
                        if exc.response.status_code == 403:
                            print(f"⚠️ Permission denied: Cannot delete Jira issue {issue_key}. Please check API token has 'Delete Issues' permission.")
                        else:
                            print(f"⚠️ Failed to delete Jira issue {issue_key}: {exc}")
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
        """
        Count pending issues from Jira project that belong to a specific tenant.
        In Jira, status is stored in description as "Status: Pending", not in the built-in status field.
        So we need to:
        1. Fetch issues by tenant_id
        2. Parse description to check for "Status: Pending"
        """
        tenant_field_id = field_map.get("tenant_id")
        if not tenant_field_id:
            raise ValueError("tenant_id field not found in field map")
        
        pending_count = 0
        start_at = 0
        
        # Use /rest/api/3/search/jql endpoint
        url = f"{self.server_url}/rest/api/3/search/jql"
        
        while True:
            # Search issues with tenant_id (status is in description, not in JQL)
            jql = f"project = {container_id} AND {tenant_field_id} ~ \"{tenant_id}\""
            
            # For /search/jql endpoint: body has jql and fields, params in query string
            payload = {
                "jql": jql,
                "fields": ["id", "key", "description", tenant_field_id]  # Get description and tenant_id field
            }
            params = {
                "startAt": start_at,
                "maxResults": batch_size
            }
            
            try:
                print(f"🔍 Fetching Jira issues (batch {start_at // batch_size + 1}):")
                print(f"   JQL: {jql}")
                print(f"   Start at: {start_at}, Max results: {batch_size}")
                
                response = requests.post(
                    url,
                    json=payload,  # JSON body with jql and fields
                    params=params,  # Other params in query string
                    headers=self._headers(),
                    timeout=20
                )
                response.raise_for_status()
                data = response.json()
                
                issues = data.get("issues", [])
                total = data.get("total", 0)
                
                print(f"   ✅ Fetched {len(issues)} issues (total: {total})")
                
                if not issues:
                    break
                
                # Check each issue
                for issue in issues:
                    issue_key = issue.get("key", "")
                    fields = issue.get("fields", {})
                    
                    # Get tenant_id from custom field
                    issue_tenant_id = None
                    tenant_field_value = fields.get(tenant_field_id)
                    if tenant_field_value:
                        # Custom field value can be string or object
                        if isinstance(tenant_field_value, str):
                            issue_tenant_id = tenant_field_value.strip()
                        elif isinstance(tenant_field_value, dict):
                            issue_tenant_id = str(tenant_field_value.get("value", "")).strip()
                        else:
                            issue_tenant_id = str(tenant_field_value).strip() if tenant_field_value else None
                    
                    # Check if tenant_id matches
                    if issue_tenant_id != tenant_id:
                        continue
                    
                    # Get description and parse status
                    description = fields.get("description")
                    description_text = ""
                    
                    # Convert ADF to text if needed
                    if description:
                        if isinstance(description, dict) and description.get("type") == "doc":
                            description_text = self._adf_to_text(description)
                        elif isinstance(description, str):
                            description_text = description
                    
                    # Parse status from description
                    # Format: "Status: Pending" or "Status:Pending"
                    status_match = re.search(r'Status:\s*([^\n]+)', description_text, re.IGNORECASE)
                    if status_match:
                        issue_status = status_match.group(1).strip()
                        print(f"   🔍 Issue {issue_key}: tenant_id={issue_tenant_id}, status={issue_status}")
                        
                        # Check if status is pending
                        if issue_status.lower() == pending_label.lower():
                            pending_count += 1
                            print(f"   ✅ Counted as pending! Total so far: {pending_count}")
                    else:
                        print(f"   ⚠️ Issue {issue_key}: No status found in description")
                
                # Check if more pages
                if start_at + len(issues) >= total:
                    break
                start_at += len(issues)
                
            except Exception as exc:
                print(f"⚠️ Failed to count Jira pending issues: {exc}")
                print(f"   URL used: {url}")
                print(f"   Server URL: {self.server_url}")
                if hasattr(exc, 'response') and exc.response is not None:
                    print(f"   Response status: {exc.response.status_code}")
                    print(f"   Response text: {exc.response.text[:200]}")
                # Return count so far instead of 0
                break
        
        print(f"📊 Total pending issues for tenant {tenant_id}: {pending_count}")
        return pending_count

    def has_pending_issues_in_description(
        self,
        container_id: str,
        tenant_id: str,
        batch_size: int = 50
    ) -> bool:
        """
        Check if a Jira project has any issues with "Email Sent: No" in description.
        
        Args:
            container_id: Jira project key
            tenant_id: Tenant ID to filter issues
            batch_size: Number of issues to fetch per batch
            
        Returns:
            True if any issue has "Email Sent: No" in description, False otherwise
        """
        try:
            # Fetch issues for the project (limited batch to check quickly)
            # Use /rest/api/3/search/jql endpoint (410 Gone error for /rest/api/3/search)
            url = f"{self.server_url}/rest/api/3/search/jql"
            jql = f"project = {container_id}"
            # For /rest/api/3/search/jql: fields parameter goes in body as array
            payload = {
                "jql": jql,
                "fields": ["id", "key", "description"]  # fields as array in body
            }
            params = {
                "startAt": 0,
                "maxResults": batch_size
            }
            
            response = requests.post(
                url,
                json=payload,
                params=params,
                headers=self._headers(),
                timeout=20
            )
            response.raise_for_status()
            data = response.json()
            issues = data.get("issues", [])
            
            # Check each issue's description for "Email Sent: No" and matching tenant_id
            for issue in issues:
                description = issue.get("fields", {}).get("description")
                if not description:
                    continue
                
                # Convert ADF to text if needed
                description_text = self._adf_to_text(description) if isinstance(description, dict) else str(description)
                
                # Check for "Email Sent: No" and tenant_id in description
                email_sent_match = re.search(r"Email Sent:\s*(Yes|No)", description_text, re.IGNORECASE)
                if email_sent_match and email_sent_match.group(1).lower() == "no":
                    # Also check if tenant_id matches (using UUID pattern)
                    tenant_pattern = rf"Tenant ID:\s*({tenant_id})"
                    if re.search(tenant_pattern, description_text, re.IGNORECASE):
                        return True
            
            return False
            
        except Exception as exc:
            print(f"⚠️ Failed to check email sent status for project {container_id}: {exc}")
            return False
    
    def _adf_to_text(self, adf: Dict) -> str:
        """
        Convert ADF (Atlassian Document Format) to plain text.
        
        Args:
            adf: ADF document structure
            
        Returns:
            Plain text string
        """
        text = ""
        if isinstance(adf, dict):
            if adf.get("type") == "doc" and adf.get("content"):
                for node in adf["content"]:
                    if node.get("type") == "paragraph" and node.get("content"):
                        for content_node in node["content"]:
                            if content_node.get("type") == "text":
                                text += content_node.get("text", "")
                        text += "\n"
                    elif node.get("type") == "text":
                        text += node.get("text", "")
                    elif node.get("content"):
                        # Recursive for nested structures
                        text += self._adf_to_text(node)
        return text.strip()

    def update_item_email_sent(
        self,
        container_id: str,
        item_id: str,
        field_map: Dict[str, str],
    ) -> Optional[dict]:
        """
        Update Email Sent status to "Yes" for a Jira issue.
        Updates the description field.
        
        Args:
            container_id: Jira project key
            item_id: Jira issue key (e.g., "PROJ-1")
            field_map: Field mapping dictionary
            
        Returns:
            Updated issue data if successful, None otherwise
        """
        try:
            # Get current issue to read description
            get_url = f"{self.server_url}/rest/api/3/issue/{item_id}"
            get_response = requests.get(get_url, headers=self._headers(), timeout=20)
            get_response.raise_for_status()
            issue_data = get_response.json()
            
            description = issue_data.get("fields", {}).get("description")
            
            # Convert ADF to text if needed
            description_text = ""
            if description:
                if isinstance(description, dict) and description.get("type") == "doc":
                    description_text = self._adf_to_text(description)
                elif isinstance(description, str):
                    description_text = description
            
            # Update Email Sent status in description
            # Replace "Email Sent: No" with "Email Sent: Yes"
            updated_description = re.sub(
                r'Email Sent:\s*(No|Yes)',
                'Email Sent: Yes',
                description_text,
                flags=re.IGNORECASE
            )
            
            # If Email Sent field not found, append it
            if not re.search(r'Email Sent:', updated_description, re.IGNORECASE):
                if updated_description and not updated_description.endswith('\n'):
                    updated_description += '\n'
                updated_description += 'Email Sent: Yes'
            
            # Convert back to ADF format
            adf_description = self._text_to_adf(updated_description)
            
            # Update issue description
            update_url = f"{self.server_url}/rest/api/3/issue/{item_id}"
            update_payload = {
                "fields": {
                    "description": adf_description
                }
            }
            
            update_response = requests.put(update_url, json=update_payload, headers=self._headers(), timeout=20)
            update_response.raise_for_status()
            
            print(f"✅ Updated Email Sent to 'Yes' for issue {item_id}")
            
            # Jira PUT requests may return empty body (204 No Content or 200 with empty body)
            # Check if response has content before parsing JSON
            if update_response.text and update_response.text.strip():
                try:
                    return update_response.json()
                except (ValueError, json.JSONDecodeError):
                    # If JSON parsing fails, return empty dict to indicate success
                    return {}
            else:
                # Empty response is normal for PUT requests - return empty dict to indicate success
                return {}
            
        except Exception as exc:
            print(f"⚠️ Failed to update Email Sent for issue {item_id}: {exc}")
            return None

    def update_items_email_sent(
        self,
        container_id: str,
        item_ids: List[str],
        field_map: Dict[str, str],
    ) -> int:
        """
        Update Email Sent status to "Yes" for multiple Jira issues.
        
        Args:
            container_id: Jira project key
            item_ids: List of Jira issue keys (e.g., ["PROJ-1", "PROJ-2"])
            field_map: Field mapping dictionary
            
        Returns:
            Number of successfully updated issues
        """
        updated_count = 0
        
        for item_id in item_ids:
            result = self.update_item_email_sent(
                container_id=container_id,
                item_id=item_id,
                field_map=field_map
            )
            # Check if result is not None (None means failure, {} or dict means success)
            if result is not None:
                updated_count += 1
        
        return updated_count

