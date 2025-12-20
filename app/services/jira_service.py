"""
Jira API Service for Scheduled Calls Integration
"""

import json
import re
import traceback
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
        Automatically creates project if it doesn't exist.
        If project_key is provided, uses that. Otherwise generates a unique key from container_name.
        """
        # Generate project key if not provided
        if not project_key:
            # Generate unique project key from container name
            # Jira project keys must be: alphanumeric only (A-Z, 0-9), uppercase, max 10 chars
            # Extract only alphanumeric characters, uppercase, max 10 chars
            alphanumeric_only = re.sub(r'[^A-Z0-9]', '', container_name.upper())
            if len(alphanumeric_only) >= 3:
                project_key = alphanumeric_only[:10]
            else:
                # If not enough chars, use email prefix or default
                # Try to extract from email if container_name contains email
                if "@" in container_name:
                    email_prefix = container_name.split("@")[0].upper()
                    email_clean = re.sub(r'[^A-Z0-9]', '', email_prefix)
                    if len(email_clean) >= 3:
                        project_key = email_clean[:10]
                    else:
                        project_key = "SCALL"  # Default fallback
                else:
                    project_key = "SCALL"  # Default fallback
        
        # Validate project key format (Jira rules: 2-10 chars, start with letter, alphanumeric only)
        if not re.match(r'^[A-Z][A-Z0-9]{1,9}$', project_key):
            raise ValueError(
                f"Invalid Jira project key format: '{project_key}'. "
                f"Project keys must be 2-10 characters, start with a letter (A-Z), and contain only uppercase letters and numbers."
            )
        
        # Check if project already exists
        check_url = f"{self.server_url}/rest/api/3/project/{project_key}"
        try:
            response = requests.get(check_url, headers=self._headers(), timeout=20)
            if response.status_code == 200:
                # Project exists, return it
                project_data = response.json()
                return {
                    "id": project_data.get("key", project_key),
                    "url": self.build_container_url(project_data.get("key", project_key)),
                }
        except requests.exceptions.RequestException:
            # Project doesn't exist, will create it
            pass
        
        # Project doesn't exist, create it
        # Try API v2 first (more stable for project creation)
        create_url_v2 = f"{self.server_url}/rest/api/2/project"
        create_url_v3 = f"{self.server_url}/rest/api/3/project"
        
        # Get current user's account ID for projectLead (required by Jira)
        project_lead_account_id = None
        try:
            # Get current user info
            user_url = f"{self.server_url}/rest/api/3/myself"
            user_response = requests.get(user_url, headers=self._headers(), timeout=20)
            if user_response.status_code == 200:
                user_data = user_response.json()
                project_lead_account_id = user_data.get("accountId")
                if project_lead_account_id:
                    print(f"✅ Got project lead account ID: {project_lead_account_id[:10]}... (from /myself)")
                else:
                    print(f"⚠️ /myself returned user data but no accountId: {user_data}")
            else:
                print(f"⚠️ /myself endpoint returned {user_response.status_code}: {user_response.text[:200]}")
                # Fallback: try to get account ID from email search
                search_url = f"{self.server_url}/rest/api/3/user/search"
                search_params = {"query": self.email}
                search_response = requests.get(search_url, headers=self._headers(), params=search_params, timeout=20)
                if search_response.status_code == 200:
                    users = search_response.json()
                    if users:
                        project_lead_account_id = users[0].get("accountId")
                        if project_lead_account_id:
                            print(f"✅ Got project lead account ID: {project_lead_account_id[:10]}... (from user search)")
                        else:
                            print(f"⚠️ User search returned users but no accountId in first user: {users[0] if users else 'No users'}")
                else:
                    print(f"⚠️ User search returned {search_response.status_code}: {search_response.text[:200]}")
        except Exception as e:
            print(f"⚠️ Warning: Could not get project lead account ID: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
        
        if not project_lead_account_id:
            raise ValueError("Could not get project lead account ID. Please ensure your Jira API token has proper permissions.")
        
        # Try to get available project templates
        project_template_key = None
        template_options = []
        
        # Try multiple template endpoints
        template_endpoints = [
            f"{self.server_url}/rest/api/3/project/templates",
            f"{self.server_url}/rest/api/3/project/type/business/accessible",
            f"{self.server_url}/rest/api/3/project/type/software/accessible"
        ]
        
        for endpoint in template_endpoints:
            try:
                templates_response = requests.get(endpoint, headers=self._headers(), timeout=20)
                if templates_response.status_code == 200:
                    templates = templates_response.json()
                    # Handle both array and object responses
                    if isinstance(templates, list):
                        template_options.extend(templates)
                    elif isinstance(templates, dict) and "values" in templates:
                        template_options.extend(templates["values"])
                    elif isinstance(templates, dict):
                        template_options.append(templates)
                    break
            except Exception as e:
                print(f"⚠️ Template endpoint {endpoint} failed: {e}")
                continue
        
        # Find suitable template
        if template_options:
            # Prefer templates with these keywords
            preferred_keywords = ["task", "basic", "kanban", "scrum", "blank", "empty"]
            for template in template_options:
                template_key = template.get("key", "") or template.get("id", "")
                template_name = template.get("name", "").lower()
                
                # Check if template key or name matches preferred keywords
                for keyword in preferred_keywords:
                    if keyword in template_key.lower() or keyword in template_name:
                        project_template_key = template_key
                        print(f"✅ Found preferred template: {project_template_key}")
                        break
                
                if project_template_key:
                    break
            
            # If no preferred template, use first available
            if not project_template_key:
                first_template = template_options[0]
                project_template_key = first_template.get("key", "") or first_template.get("id", "")
                print(f"✅ Using first available template: {project_template_key}")
        
        if not project_template_key:
            print(f"⚠️ Warning: No project templates found. Will try without template.")
        
        # Create project payload - Try both API v2 and v3 formats
        # API v2 is more stable for project creation
        payloads_to_try = []
        
        # API v2 formats - Try WITHOUT lead first (GDPR strict mode issue)
        # Format 1: v2 Business type WITHOUT lead (let Jira assign default)
        payloads_to_try.append({
            "key": project_key,
            "name": container_name,
            "projectTypeKey": "business"
        })
        
        # Format 2: v2 Software type WITHOUT lead
        payloads_to_try.append({
            "key": project_key,
            "name": container_name,
            "projectTypeKey": "software"
        })
        
        # Format 3: v2 Minimal WITHOUT lead and projectTypeKey
        payloads_to_try.append({
            "key": project_key,
            "name": container_name
        })
        
        # Format 4: v2 Business WITH lead (fallback if without lead fails)
        payloads_to_try.append({
            "key": project_key,
            "name": container_name,
            "projectTypeKey": "business",
            "lead": project_lead_account_id
        })
        
        # Format 5: v2 Software WITH lead
        payloads_to_try.append({
            "key": project_key,
            "name": container_name,
            "projectTypeKey": "software",
            "lead": project_lead_account_id
        })
        
        # API v3 formats - Try WITHOUT projectLead first
        # Format 6: v3 Business type WITHOUT projectLead
        payloads_to_try.append({
            "key": project_key,
            "name": container_name,
            "projectTypeKey": "business"
        })
        
        # Format 7: v3 Software type WITHOUT projectLead
        payloads_to_try.append({
            "key": project_key,
            "name": container_name,
            "projectTypeKey": "software"
        })
        
        # Format 8: v3 Business WITH projectLead (fallback)
        payloads_to_try.append({
            "key": project_key,
            "name": container_name,
            "projectTypeKey": "business",
            "projectLead": {"accountId": project_lead_account_id}
        })
        
        # Format 9: v3 Software WITH projectLead
        payloads_to_try.append({
            "key": project_key,
            "name": container_name,
            "projectTypeKey": "software",
            "projectLead": {"accountId": project_lead_account_id}
        })
        
        # Format 10-12: With templates (only if template exists)
        if project_template_key:
            # Format 10: v2 with template WITHOUT lead
            payloads_to_try.append({
                "key": project_key,
                "name": container_name,
                "projectTypeKey": "business",
                "projectTemplateKey": project_template_key
            })
            
            # Format 11: v3 with template WITHOUT projectLead
            payloads_to_try.append({
                "key": project_key,
                "name": container_name,
                "projectTypeKey": "business",
                "projectTemplateKey": project_template_key
            })
            
            # Format 12: v3 with template WITH projectLead (fallback)
            payloads_to_try.append({
                "key": project_key,
                "name": container_name,
                "projectTypeKey": "business",
                "projectTemplateKey": project_template_key,
                "projectLead": {"accountId": project_lead_account_id}
            })
        
        last_error = None
        total_formats = len(payloads_to_try)
        for idx, payload in enumerate(payloads_to_try, 1):
            try:
                # Determine which endpoint to use based on payload format
                # v2 uses "lead", v3 uses "projectLead"
                if "lead" in payload:
                    create_url = create_url_v2
                    api_version = "v2"
                else:
                    create_url = create_url_v3
                    api_version = "v3"
                
                # Log payload for debugging (mask accountId/lead for security)
                payload_log = payload.copy()
                if "lead" in payload_log:
                    payload_log["lead"] = "***masked***"
                elif "projectLead" in payload_log:
                    if isinstance(payload_log["projectLead"], dict):
                        payload_log["projectLead"] = {"accountId": "***masked***"}
                    else:
                        payload_log["projectLead"] = "***masked***"
                
                print(f"🔍 Trying Jira project creation format {idx}/{total_formats} (API {api_version}) with key: {project_key}")
                print(f"   Payload: {json.dumps(payload_log, indent=2)}")
                
                response = requests.post(create_url, json=payload, headers=self._headers(), timeout=30)
                response.raise_for_status()
                project_data = response.json()
                
                created_key = project_data.get("key", project_key)
                print(f"✅ Successfully created Jira project: {created_key} (using API {api_version})")
                return {
                    "id": created_key,
                    "url": self.build_container_url(created_key),
                }
            except requests.exceptions.HTTPError as e:
                last_error = e
                # Log error for debugging
                try:
                    error_data = e.response.json()
                    error_msgs = error_data.get('errorMessages', [])
                    errors_dict = error_data.get('errors', {})
                    print(f"❌ Format {idx} failed: {error_msgs}")
                    if errors_dict:
                        print(f"   Detailed errors: {errors_dict}")
                    # Log response for debugging
                    print(f"   Response status: {e.response.status_code}")
                    print(f"   Response body: {e.response.text[:300]}")
                except:
                    print(f"❌ Format {idx} failed: {e.response.status_code}")
                    print(f"   Response: {e.response.text[:300]}")
                
                # If 400 error, try next format
                if e.response.status_code == 400:
                    continue
                # For other errors (403, etc.), break and handle
                break
            except Exception as e:
                last_error = e
                print(f"❌ Format {idx} exception: {str(e)}")
                continue
        
        # If all formats failed, handle the last error
        if not last_error:
            raise ValueError("Failed to create Jira project: No error information available")
        
        if isinstance(last_error, requests.exceptions.HTTPError):
            e = last_error
            # If project creation fails (e.g., permissions), try to use existing project
            if e.response.status_code == 403:
                raise ValueError(f"Permission denied: Cannot create Jira project. Please create project '{project_key}' manually or provide an existing project_key.")
            elif e.response.status_code == 400:
                try:
                    error_data = e.response.json()
                    errors = error_data.get("errors", {})
                    error_messages = error_data.get("errorMessages", [])
                    
                    # Check if project key already exists
                    if "projectKey" in errors:
                        # Project key already exists or invalid, try to get it
                        try:
                            get_response = requests.get(check_url, headers=self._headers(), timeout=20)
                            if get_response.status_code == 200:
                                existing_project = get_response.json()
                                return {
                                    "id": existing_project.get("key", project_key),
                                    "url": self.build_container_url(existing_project.get("key", project_key)),
                                }
                        except:
                            pass
                    
                    # Build error message
                    error_msg = ""
                    if errors:
                        error_msg = f"Errors: {errors}"
                    if error_messages:
                        error_msg += f" Messages: {', '.join(error_messages)}"
                    if not error_msg:
                        error_msg = f"Response: {e.response.text[:500]}"
                    
                    # Provide helpful suggestion with manual creation steps
                    suggestion = f"\n\n💡 Jira project creation via API failed. This usually requires 'Administer Jira' permission."
                    suggestion += f"\n\n📋 Manual Creation Steps:"
                    suggestion += f"\n   1. Go to: {self.server_url}/secure/project/CreateProject!default.jspa"
                    suggestion += f"\n   2. Create project with:"
                    suggestion += f"\n      - Project Key: {project_key}"
                    suggestion += f"\n      - Project Name: {container_name}"
                    suggestion += f"\n   3. After creation, the system will automatically use this project."
                    suggestion += f"\n\n🔑 Alternative: Ensure your API token has 'Administer Jira' permission."
                    suggestion += f"\n   Token email: {self.email}"
                    
                    raise ValueError(f"Failed to create Jira project after trying all {total_formats} formats: {error_msg}{suggestion}")
                except (ValueError, KeyError):
                    # If JSON parsing fails, use raw response
                    raise ValueError(f"Failed to create Jira project: {e.response.text[:500]}")
            raise ValueError(f"Failed to create Jira project: {e.response.text[:500]}")
        else:
            raise ValueError(f"Failed to create Jira project: {str(last_error)}")

    def ensure_required_fields(self, container_id: str) -> Dict[str, str]:
        """
        Ensure Jira project has required custom fields.
        Automatically creates missing fields.
        """
        field_map = {}
        errors = []  # Track errors for debugging
        
        try:
            # Get all fields (both built-in and custom)
            url = f"{self.server_url}/rest/api/3/field"
            print(f"🔍 Fetching Jira fields from: {url}")
            
            try:
                response = requests.get(url, headers=self._headers(), timeout=20)
                response.raise_for_status()
                all_fields = response.json()
                print(f"✅ Successfully fetched {len(all_fields)} Jira fields")
            except requests.exceptions.HTTPError as e:
                error_detail = {
                    "operation": "fetch_fields",
                    "url": url,
                    "status_code": e.response.status_code if e.response else None,
                    "response": e.response.text[:500] if e.response else str(e),
                    "error": str(e)
                }
                errors.append(error_detail)
                print(f"❌ Failed to fetch Jira fields:")
                print(f"   Status: {error_detail['status_code']}")
                print(f"   Response: {error_detail['response']}")
                print(f"   Error: {error_detail['error']}")
                # Return empty field_map if we can't fetch fields
                return field_map
            except requests.exceptions.RequestException as e:
                error_detail = {
                    "operation": "fetch_fields",
                    "url": url,
                    "error_type": type(e).__name__,
                    "error": str(e)
                }
                errors.append(error_detail)
                print(f"❌ Network error fetching Jira fields: {error_detail['error']}")
                return field_map
            except Exception as e:
                error_detail = {
                    "operation": "fetch_fields",
                    "url": url,
                    "error_type": type(e).__name__,
                    "error": str(e)
                }
                errors.append(error_detail)
                print(f"❌ Unexpected error fetching Jira fields: {error_detail['error']}")
                return field_map
            
            # Map fields by name
            field_by_name = {f.get("name", "").lower(): f for f in all_fields}
            
            # Jira field type and searcher mapping
            jira_field_config = {
                "text": {
                    "type": "com.atlassian.jira.plugin.system.customfieldtypes:textfield",
                    "searcherKey": "com.atlassian.jira.plugin.system.customfieldtypes:textsearcher"
                },
                "select": {
                    "type": "com.atlassian.jira.plugin.system.customfieldtypes:select",
                    "searcherKey": "com.atlassian.jira.plugin.system.customfieldtypes:selectsearcher"
                }
            }
            
            # Check for required fields and create if missing
            for field_def in self.REQUIRED_FIELDS:
                field_name = field_def["title"]
                field_key = field_def["key"]
                field_type = field_def["type"]
                
                try:
                    # Status is a built-in field
                    if field_key == "status":
                        field_map[field_key] = "status"
                        print(f"✅ Using built-in Jira field: {field_name}")
                        continue
                    
                    # Check if field already exists
                    if field_name.lower() in field_by_name:
                        field_id = field_by_name[field_name.lower()].get("id", "")
                        if field_id:
                            field_map[field_key] = field_id
                            print(f"✅ Found existing Jira field: {field_name} (ID: {field_id})")
                            continue
                        else:
                            print(f"⚠️ Field {field_name} found but has no ID")
                    
                    # Field doesn't exist, create it
                    config = jira_field_config.get(field_type, jira_field_config["text"])
                    
                    # Create the custom field
                    field_payload = {
                        "name": field_name,
                        "type": config["type"],
                        "searcherKey": config["searcherKey"],
                        "description": f"Custom field for {field_name}"
                    }
                    
                    print(f"🔍 Creating Jira field {field_name} (type: {field_type})...")
                    print(f"   Field payload: {json.dumps(field_payload, indent=2)}")
                    
                    create_url = f"{self.server_url}/rest/api/3/field"
                    
                    try:
                        create_response = requests.post(create_url, json=field_payload, headers=self._headers(), timeout=20)
                        
                        if create_response.status_code in [200, 201]:
                            try:
                                created_field = create_response.json()
                                field_id = created_field.get("id", "")
                                
                                if field_id:
                                    field_map[field_key] = field_id
                                    print(f"✅ Created Jira field {field_name} with ID: {field_id}")
                                else:
                                    error_detail = {
                                        "operation": "create_field",
                                        "field_name": field_name,
                                        "field_key": field_key,
                                        "status_code": create_response.status_code,
                                        "response": create_response.text[:500],
                                        "error": "Field created but no ID returned"
                                    }
                                    errors.append(error_detail)
                                    print(f"⚠️ Jira field {field_name} created but no ID returned")
                                    print(f"   Response: {create_response.text[:500]}")
                            except (ValueError, KeyError) as e:
                                error_detail = {
                                    "operation": "parse_field_response",
                                    "field_name": field_name,
                                    "status_code": create_response.status_code,
                                    "response": create_response.text[:500],
                                    "error": str(e),
                                    "error_type": type(e).__name__
                                }
                                errors.append(error_detail)
                                print(f"⚠️ Failed to parse Jira field creation response for {field_name}: {str(e)}")
                                print(f"   Response: {create_response.text[:500]}")
                        else:
                            # HTTP error
                            try:
                                error_data = create_response.json()
                                error_messages = error_data.get("errorMessages", [])
                                errors_dict = error_data.get("errors", {})
                            except:
                                error_messages = []
                                errors_dict = {}
                            
                            error_detail = {
                                "operation": "create_field",
                                "field_name": field_name,
                                "field_key": field_key,
                                "field_type": field_type,
                                "url": create_url,
                                "status_code": create_response.status_code,
                                "response": create_response.text[:500],
                                "error_messages": error_messages,
                                "errors": errors_dict,
                                "payload": field_payload
                            }
                            errors.append(error_detail)
                            print(f"❌ Failed to create Jira field {field_name}:")
                            print(f"   Status: {create_response.status_code}")
                            print(f"   Error Messages: {error_messages}")
                            print(f"   Errors: {errors_dict}")
                            print(f"   Response: {create_response.text[:500]}")
                            print(f"   Payload: {json.dumps(field_payload, indent=2)}")
                    except requests.exceptions.HTTPError as e:
                        error_detail = {
                            "operation": "create_field",
                            "field_name": field_name,
                            "field_key": field_key,
                            "url": create_url,
                            "status_code": e.response.status_code if e.response else None,
                            "response": e.response.text[:500] if e.response else str(e),
                            "error": str(e),
                            "payload": field_payload
                        }
                        errors.append(error_detail)
                        print(f"❌ HTTP error creating Jira field {field_name}: {str(e)}")
                        if e.response:
                            print(f"   Status: {e.response.status_code}")
                            print(f"   Response: {e.response.text[:500]}")
                    except requests.exceptions.RequestException as e:
                        error_detail = {
                            "operation": "create_field",
                            "field_name": field_name,
                            "field_key": field_key,
                            "url": create_url,
                            "error_type": type(e).__name__,
                            "error": str(e),
                            "payload": field_payload
                        }
                        errors.append(error_detail)
                        print(f"❌ Network error creating Jira field {field_name}: {str(e)}")
                    except Exception as e:
                        error_detail = {
                            "operation": "create_field",
                            "field_name": field_name,
                            "field_key": field_key,
                            "error_type": type(e).__name__,
                            "error": str(e),
                            "payload": field_payload
                        }
                        errors.append(error_detail)
                        print(f"❌ Unexpected error creating Jira field {field_name}: {str(e)}")
                        import traceback
                        print(f"   Traceback: {traceback.format_exc()}")
                except Exception as e:
                    error_detail = {
                        "operation": "process_field",
                        "field_name": field_name,
                        "field_key": field_key,
                        "error_type": type(e).__name__,
                        "error": str(e)
                    }
                    errors.append(error_detail)
                    print(f"❌ Error processing field {field_name}: {str(e)}")
                    print(f"   Traceback: {traceback.format_exc()}")
        
        except Exception as e:
            error_detail = {
                "operation": "ensure_required_fields",
                "container_id": container_id,
                "error_type": type(e).__name__,
                "error": str(e)
            }
            errors.append(error_detail)
            print(f"❌ Critical error in ensure_required_fields: {str(e)}")
            print(f"   Traceback: {traceback.format_exc()}")
        finally:
            # Log summary
            if errors:
                print(f"\n📋 Error Summary ({len(errors)} errors):")
                for idx, err in enumerate(errors, 1):
                    print(f"   {idx}. {err.get('operation', 'unknown')} - {err.get('error', 'unknown error')}")
            print(f"📊 Jira field_map: {field_map} (Total fields: {len(field_map)})")
        
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
        # Status: Set to "Pending" by default
        if "status" in field_map:
            fields["status"] = {"name": "Pending"}
        
        # Email Sent: Set to "No" by default
        if "email_sent" in field_map:
            fields[field_map["email_sent"]] = {"value": "No"}
        
        # Other custom fields
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
        if "call_session_id" in field_map:
            # Leave blank initially (will be updated later when call is initiated)
            fields[field_map["call_session_id"]] = ""
        
        payload = {"fields": fields}
        
        try:
            print(f"🔍 Creating Jira issue for {phone_number} in project {container_id}...")
            print(f"   URL: {url}")
            print(f"   Payload: {json.dumps(payload, indent=2)}")
            
            response = requests.post(url, json=payload, headers=self._headers(), timeout=20)
            
            if response.status_code in [200, 201]:
                issue_data = response.json()
                issue_key = issue_data.get("key", "")
                issue_id = issue_data.get("id", "")
                print(f"✅ Successfully created Jira issue {issue_key} (ID: {issue_id}) for {phone_number}")
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
                    "payload": payload
                }
                
                print(f"❌ Failed to create Jira issue for {phone_number}:")
                print(f"   Status: {response.status_code}")
                print(f"   Error Messages: {error_messages}")
                print(f"   Errors: {errors_dict}")
                print(f"   Response: {response.text[:500]}")
                print(f"   Payload: {json.dumps(payload, indent=2)}")
                
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
                "payload": payload
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
                "payload": payload
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
                "payload": payload
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

