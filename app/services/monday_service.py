"""
Monday.com API Service for Scheduled Calls Integration
"""

import json
from typing import Dict, List, Optional, Tuple

import requests

from app.core.config import settings
from app.services.base_crm_service import BaseCRMService
from app.core.security import decrypt_api_key


class MondayService(BaseCRMService):
    """Service for interacting with Monday.com API"""

    API_URL = "https://api.monday.com/v2"
    REQUIRED_COLUMNS = [
        {
            "key": "status",
            "title": "Status",
            "type": "status",
            "defaults": {"labels": {"0": "Pending", "1": "Called", "2": "Failed"}},
        },
        {"key": "agent_id", "title": "Agent ID", "type": "text"},
        {"key": "call_time_utc", "title": "Call Time UTC", "type": "text"},
        {"key": "tenant_id", "title": "Tenant ID", "type": "text"},
        {"key": "user_id", "title": "User ID", "type": "text"},
        {"key": "batch_id", "title": "Batch ID", "type": "text"},
        {"key": "call_session_id", "title": "Call Session ID", "type": "text"},
        {"key": "phone_number_id", "title": "Phone Number ID", "type": "text"},  # ✅ Optional phone number ID
        {
            "key": "email_sent",
            "title": "Email Sent",
            "type": "status",
            "defaults": {"labels": {"0": "No", "1": "Yes"}},
        },
    ]

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize MondayService with optional API key.
        If not provided, uses settings.MONDAY_API_KEY (for backward compatibility).
        """
        self._api_key = api_key

    def get_api_key(self) -> str:
        """Get decrypted API key"""
        if not self._api_key or self._api_key.strip() == "":
            # Fallback to ENV key if instance key not provided
            env_key = settings.MONDAY_API_KEY or ""
            if env_key:
                print(f"⚠️ Using ENV Monday.com API key (instance key not provided)")
            return env_key
        
        # Check if encrypted (JWT format)
        if self._api_key.startswith("eyJ"):
            try:
                decrypted = decrypt_api_key(self._api_key)
                
                # Debug logging
                print(f"🔍 Monday.com API Key decrypted successfully")
                print(f"   Encrypted (first 20 chars): {self._api_key[:20]}...")
                print(f"   Decrypted (first 10 chars): {decrypted[:10] if decrypted else 'None'}...")
                print(f"   Decrypted length: {len(decrypted) if decrypted else 0}")
                
                if not decrypted or decrypted.strip() == "":
                    raise ValueError("Decrypted API key is empty")
                return decrypted
            except Exception as exc:
                print(f"❌ Monday.com API key decryption failed: {str(exc)}")
                import traceback
                traceback.print_exc()
                raise ValueError(f"Failed to decrypt Monday.com API key: {str(exc)}")
        
        # Already decrypted or plain text
        print(f"🔍 Monday.com API Key appears to be already decrypted")
        return self._api_key

    def build_container_url(self, container_id: str) -> str:
        """Build URL for Monday.com board (implements BaseCRMService)"""
        return self.build_board_url(container_id)

    @staticmethod
    def build_board_url(board_id: str) -> str:
        return f"https://app.monday.com/boards/{board_id}"

    def _headers(self) -> Dict[str, str]:
        """Get API headers (instance method)"""
        api_key = self.get_api_key()
        if not api_key or api_key.strip() == "":
            raise ValueError("Monday.com API key is not configured or is empty")
        
        # Validate API key format (Monday.com keys are usually long)
        if len(api_key) < 10:
            raise ValueError(f"Monday.com API key seems too short: {len(api_key)} chars. Minimum expected: 10 chars")
        
        return {
            "Authorization": api_key,
            "Content-Type": "application/json",
            "API-Version": "2024-01",
        }

    @staticmethod
    def _headers_static() -> Dict[str, str]:
        """Get API headers (static method for backward compatibility)"""
        if not settings.MONDAY_API_KEY:
            raise ValueError("Monday.com API key is not configured")
        return {
            "Authorization": settings.MONDAY_API_KEY,
            "Content-Type": "application/json",
            "API-Version": "2024-01",
        }

    def _execute(self, query: str, variables: Dict) -> Dict:
        """Execute GraphQL query (instance method)"""
        print(f"🔍 MondayService._execute called")
        print(f"   API URL: {self.API_URL}")
        
        try:
            headers = self._headers()
            print(f"   Headers prepared (API key length: {len(headers.get('Authorization', ''))})")
            print(f"   API key first 10 chars: {headers.get('Authorization', '')[:10]}...")
        except Exception as header_exc:
            print(f"❌ Failed to get headers: {header_exc}")
            import traceback
            traceback.print_exc()
            raise
        
        try:
            print(f"   Making request to Monday.com API...")
            response = requests.post(
                self.API_URL,
                json={"query": query, "variables": variables},
                headers=headers,
                timeout=20,
            )
            print(f"   Response status: {response.status_code}")
            
            response.raise_for_status()
            payload = response.json()
            
            if "errors" in payload:
                print(f"❌ Monday.com API returned errors: {payload['errors']}")
                raise ValueError(payload["errors"])
            
            print(f"✅ Monday.com API request successful")
            return payload.get("data", {})
        except requests.exceptions.HTTPError as http_exc:
            print(f"❌ HTTP Error: {http_exc}")
            print(f"   Status code: {http_exc.response.status_code if hasattr(http_exc, 'response') else 'unknown'}")
            if hasattr(http_exc, 'response') and http_exc.response is not None:
                try:
                    error_body = http_exc.response.text
                    print(f"   Error response: {error_body[:500]}")
                except:
                    pass
            raise
        except Exception as exc:
            print(f"❌ Unexpected error in _execute: {exc}")
            import traceback
            traceback.print_exc()
            raise

    @staticmethod
    def _execute_static(query: str, variables: Dict) -> Dict:
        """Execute GraphQL query (static method for backward compatibility)"""
        response = requests.post(
            MondayService.API_URL,
            json={"query": query, "variables": variables},
            headers=MondayService._headers_static(),
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        if "errors" in payload:
            raise ValueError(payload["errors"])
        return payload.get("data", {})

    def create_container(self, container_name: str, **kwargs) -> Dict[str, str]:
        """Create a Monday.com board (implements BaseCRMService)"""
        workspace_id = kwargs.get("workspace_id")
        return self.create_board(container_name, workspace_id)

    def create_board(self, board_name: str, workspace_id: Optional[str] = None) -> Dict[str, str]:
        """
        Create a dedicated Monday.com board for a tenant (instance method).
        """
        query = """
        mutation ($boardName: String!, $workspaceId: ID) {
            create_board (board_name: $boardName, board_kind: private, workspace_id: $workspaceId) {
                id
                name
            }
        }
        """
        variables: Dict[str, Optional[str]] = {"boardName": board_name, "workspaceId": workspace_id}

        data = self._execute(query, variables)
        board = data.get("create_board")
        if not board:
            raise ValueError("Failed to create Monday.com board")

        board_id = str(board["id"])
        return {"id": board_id, "url": self.build_container_url(board_id)}

    @staticmethod
    def create_board_static(board_name: str, workspace_id: Optional[str] = None) -> Dict[str, str]:
        """
        Create a dedicated Monday.com board for a tenant (static method for backward compatibility).
        """
        query = """
        mutation ($boardName: String!, $workspaceId: ID) {
            create_board (board_name: $boardName, board_kind: private, workspace_id: $workspaceId) {
                id
                name
            }
        }
        """
        variables: Dict[str, Optional[str]] = {"boardName": board_name, "workspaceId": workspace_id}

        data = MondayService._execute_static(query, variables)
        board = data.get("create_board")
        if not board:
            raise ValueError("Failed to create Monday.com board")

        board_id = str(board["id"])
        return {"id": board_id, "url": MondayService.build_board_url(board_id)}

    def get_board_columns(self, board_id: str) -> List[Dict]:
        """Get board columns (instance method - uses DB API key)"""
        query = """
        query ($boardId: [ID!]) {
            boards (ids: $boardId) {
                columns {
                    id
                    title
                    type
                }
            }
        }
        """
        data = self._execute(query, {"boardId": board_id})
        boards = data.get("boards") or []
        if not boards:
            raise ValueError(f"Board {board_id} not found")
        return boards[0].get("columns", [])
    
    @staticmethod
    def get_board_columns_static(board_id: str) -> List[Dict]:
        """Get board columns (static method for backward compatibility)"""
        query = """
        query ($boardId: [ID!]) {
            boards (ids: $boardId) {
                columns {
                    id
                    title
                    type
                }
            }
        }
        """
        data = MondayService._execute_static(query, {"boardId": board_id})
        boards = data.get("boards") or []
        if not boards:
            raise ValueError(f"Board {board_id} not found")
        return boards[0].get("columns", [])

    def create_column(self, board_id: str, title: str, column_type: str, defaults: Optional[Dict] = None) -> Dict:
        """Create column (instance method - uses DB API key)"""
        query = """
        mutation ($boardId: ID!, $title: String!, $type: ColumnType!, $defaults: JSON) {
            create_column (board_id: $boardId, title: $title, column_type: $type, defaults: $defaults) {
                id
                title
                type
            }
        }
        """
        # Monday.com expects defaults as a JSON string, not a dict
        defaults_json = json.dumps(defaults) if defaults else None
        
        variables = {
            "boardId": board_id,
            "title": title,
            "type": column_type,
            "defaults": defaults_json,
        }
        data = self._execute(query, variables)
        column = data.get("create_column")
        if not column:
            raise ValueError(f"Failed to create column {title} on board {board_id}")
        return column
    
    @staticmethod
    def create_column_static(board_id: str, title: str, column_type: str, defaults: Optional[Dict] = None) -> Dict:
        """Create column (static method for backward compatibility)"""
        query = """
        mutation ($boardId: ID!, $title: String!, $type: ColumnType!, $defaults: JSON) {
            create_column (board_id: $boardId, title: $title, column_type: $type, defaults: $defaults) {
                id
                title
                type
            }
        }
        """
        # Monday.com expects defaults as a JSON string, not a dict
        defaults_json = json.dumps(defaults) if defaults else None
        
        variables = {
            "boardId": board_id,
            "title": title,
            "type": column_type,
            "defaults": defaults_json,
        }
        data = MondayService._execute_static(query, variables)
        column = data.get("create_column")
        if not column:
            raise ValueError(f"Failed to create column {title} on board {board_id}")
        return column

    def delete_column(self, board_id: str, column_id: str) -> None:
        """Delete column (instance method - uses DB API key)"""
        query = """
        mutation ($boardId: ID!, $columnId: String!) {
            delete_column (board_id: $boardId, column_id: $columnId) {
                id
            }
        }
        """
        self._execute(query, {"boardId": board_id, "columnId": column_id})
    
    @staticmethod
    def delete_column_static(board_id: str, column_id: str) -> None:
        """Delete column (static method for backward compatibility)"""
        query = """
        mutation ($boardId: ID!, $columnId: String!) {
            delete_column (board_id: $boardId, column_id: $columnId) {
                id
            }
        }
        """
        MondayService._execute_static(query, {"boardId": board_id, "columnId": column_id})

    @staticmethod
    def ensure_required_columns(board_id: str) -> Dict[str, str]:
        """
        Ensure the scheduled-calls board has exactly the required columns.
        Static method for backward compatibility (uses settings.MONDAY_API_KEY).

        Returns:
            Dict mapping required column keys to their Monday column IDs.
        """
        required_titles = {c["title"].lower() for c in MondayService.REQUIRED_COLUMNS}
        columns = MondayService.get_board_columns_static(board_id)
        type_lookup = {col_def["title"].lower(): col_def["type"] for col_def in MondayService.REQUIRED_COLUMNS}

        # Remove required-title columns with mismatched types to avoid duplicates
        for col in columns:
            title = col["title"].lower()
            if title in required_titles and col.get("type") != type_lookup.get(title):
                try:
                    MondayService.delete_column_static(board_id, col["id"])
                except Exception as exc:
                    print(f"⚠️ Failed to remove mismatched column {col['id']} on board {board_id}: {exc}")

        # Refresh columns after cleanup
        columns = MondayService.get_board_columns_static(board_id)
        title_lookup = {col["title"].lower(): col for col in columns}

        # Create missing required columns
        for col_def in MondayService.REQUIRED_COLUMNS:
            current = title_lookup.get(col_def["title"].lower())
            if not current:
                created = MondayService.create_column_static(
                    board_id=board_id,
                    title=col_def["title"],
                    column_type=col_def["type"],
                    defaults=col_def.get("defaults"),
                )
                title_lookup[col_def["title"].lower()] = created

        # Refresh columns to get final IDs
        columns = MondayService.get_board_columns_static(board_id)
        title_lookup = {col["title"].lower(): col for col in columns}

        # Remove extraneous columns (keep item name)
        for col in columns:
            title = col["title"].lower()
            if col.get("type") == "name":
                continue
            if title not in required_titles:
                try:
                    MondayService.delete_column_static(board_id, col["id"])
                except Exception as exc:
                    print(f"⚠️ Failed to delete extra column {col['id']} on board {board_id}: {exc}")

        # Final mapping
        required_map: Dict[str, str] = {}
        for col_def in MondayService.REQUIRED_COLUMNS:
            match = title_lookup.get(col_def["title"].lower())
            if not match:
                raise ValueError(f"Missing required column {col_def['title']} on board {board_id}")
            required_map[col_def["key"]] = match["id"]

        return required_map

    @staticmethod
    def create_scheduled_call_item(
        board_id: str,
        column_map: Dict[str, str],
        phone_number: str,
        agent_id: str,
        call_time_utc: str,
        tenant_id: str,
        user_id: str,
        batch_id: Optional[str] = None,
        phone_number_id: Optional[str] = None,  # ✅ Add phone_number_id parameter
    ) -> Optional[dict]:
        """Create a scheduled call item in the tenant's Monday.com board."""
        required_keys = {"status", "agent_id", "call_time_utc", "tenant_id", "user_id"}
        missing = required_keys - set(column_map.keys())
        if missing:
            raise ValueError(f"Missing Monday column ids for: {', '.join(sorted(missing))}")

        column_values = {
            column_map["status"]: {"label": "Pending"},
            column_map["agent_id"]: agent_id,
            column_map["call_time_utc"]: call_time_utc,
            column_map["tenant_id"]: tenant_id,
            column_map["user_id"]: user_id,
        }
        
        # Add batch_id if provided and column exists
        if batch_id and "batch_id" in column_map:
            column_values[column_map["batch_id"]] = batch_id
        
        # Add phone_number_id if provided and column exists
        if phone_number_id and "phone_number_id" in column_map:
            column_values[column_map["phone_number_id"]] = phone_number_id
        
        # Set Email Sent to "No" by default if column exists
        if "email_sent" in column_map:
            column_values[column_map["email_sent"]] = {"label": "No"}

        query = """
        mutation ($boardId: ID!, $itemName: String!, $columnValues: JSON!) {
            create_item (
                board_id: $boardId,
                item_name: $itemName,
                column_values: $columnValues
            ) {
                id
                name
            }
        }
        """
        variables = {
            "boardId": board_id,
            "itemName": phone_number,
            "columnValues": json.dumps(column_values),
        }

        try:
            data = MondayService._execute_static(query, variables)
            return data.get("create_item")
        except Exception as exc:
            print(f"⚠️ Failed to create Monday.com item for {phone_number}: {exc}")
            return None

    @staticmethod
    def update_item_status(
        item_id: str,
        status: str,
        board_id: Optional[str],
        column_map: Optional[Dict[str, str]] = None,
    ) -> Optional[dict]:
        """Update the status column for a Monday.com item."""
        target_board_id = board_id or settings.MONDAY_BOARD_ID
        if not target_board_id:
            return None

        if column_map is None:
            column_map = MondayService.ensure_required_columns(target_board_id)

        status_column_id = column_map.get("status", "status")
        column_values = {status_column_id: {"label": status}}

        query = """
        mutation ($boardId: ID!, $itemId: ID!, $columnValues: JSON!) {
            change_multiple_column_values (
                board_id: $boardId,
                item_id: $itemId,
                column_values: $columnValues
            ) {
                id
            }
        }
        """

        variables = {
            "boardId": target_board_id,
            "itemId": item_id,
            "columnValues": json.dumps(column_values),
        }

        try:
            return MondayService._execute_static(query, variables)
        except Exception as exc:
            print(f"⚠️ Failed to update Monday.com item {item_id}: {exc}")
            return None

    @staticmethod
    def _fetch_item_page(board_id: str, cursor: Optional[str], limit: int) -> Tuple[List[str], Optional[str]]:
        query = """
        query ($boardId: [ID!], $cursor: String, $limit: Int!) {
            boards (ids: $boardId) {
                items_page (cursor: $cursor, limit: $limit) {
                    cursor
                    items { id }
                }
            }
        }
        """
        data = MondayService._execute(query, {"boardId": board_id, "cursor": cursor, "limit": limit})
        boards = data.get("boards") or []
        if not boards:
            return [], None
        page = boards[0].get("items_page") or {}
        items = page.get("items") or []
        next_cursor = page.get("cursor")
        return [item["id"] for item in items], next_cursor

    @staticmethod
    def delete_item(item_id: str) -> None:
        query = """
        mutation ($itemId: ID!) {
            delete_item (item_id: $itemId) { id }
        }
        """
        MondayService._execute_static(query, {"itemId": item_id})

    @staticmethod
    def delete_all_items(board_id: str, batch_size: int = 50) -> int:
        """
        Delete all items from a board while keeping the board and columns intact.

        Returns:
            Number of items deleted.
        """
        deleted = 0
        cursor: Optional[str] = None

        while True:
            item_ids, cursor = MondayService._fetch_item_page(board_id, cursor, batch_size)
            if not item_ids:
                break

            for item_id in item_ids:
                try:
                    MondayService.delete_item(item_id)
                    deleted += 1
                except Exception as exc:
                    print(f"⚠️ Failed to delete Monday.com item {item_id}: {exc}")

            if not cursor:
                break

        return deleted

    @staticmethod
    def _fetch_items_with_columns(board_id: str, cursor: Optional[str], limit: int, column_ids: List[str]) -> Tuple[List[Dict], Optional[str]]:
        """Fetch items with specific column values for filtering."""
        query = """
        query ($boardId: [ID!], $cursor: String, $limit: Int!, $columnIds: [String!]) {
            boards (ids: $boardId) {
                items_page (cursor: $cursor, limit: $limit) {
                    cursor
                    items {
                        id
                        name
                        column_values(ids: $columnIds) {
                            id
                            text
                        }
                    }
                }
            }
        }
        """
        data = MondayService._execute_static(query, {
            "boardId": board_id,
            "cursor": cursor,
            "limit": limit,
            "columnIds": column_ids
        })
        boards = data.get("boards") or []
        if not boards:
            return [], None
        page = boards[0].get("items_page") or {}
        items = page.get("items") or []
        next_cursor = page.get("cursor")
        return items, next_cursor

    def delete_items_by_tenant(
        self,
        container_id: str,
        tenant_id: str,
        field_map: Dict[str, str],
        batch_size: int = 50
    ) -> int:
        """Delete items by tenant (implements BaseCRMService)"""
        return MondayService.delete_items_by_tenant_static(container_id, tenant_id, field_map, batch_size)

    @staticmethod
    def delete_items_by_tenant_static(board_id: str, tenant_id: str, column_map: Dict[str, str], batch_size: int = 50) -> int:
        """
        Delete items from board that belong to a specific tenant.
        Filters by tenant_id column.

        Args:
            board_id: Monday.com board ID
            tenant_id: Tenant ID to filter by (UUID string)
            column_map: Column mapping dictionary (must include "tenant_id")
            batch_size: Number of items to fetch per batch

        Returns:
            Number of items deleted.
        """
        tenant_column_id = column_map.get("tenant_id")
        if not tenant_column_id:
            raise ValueError("tenant_id column not found in board column map")

        deleted = 0
        cursor: Optional[str] = None

        while True:
            # Fetch items with tenant_id column
            items, cursor = MondayService._fetch_items_with_columns(
                board_id=board_id,
                cursor=cursor,
                limit=batch_size,
                column_ids=[tenant_column_id]
            )

            if not items:
                break

            for item in items:
                # Check if item belongs to this tenant
                item_tenant_id = None
                for col_val in item.get("column_values", []):
                    if col_val.get("id") == tenant_column_id:
                        item_tenant_id = col_val.get("text", "").strip()
                        break

                # Delete if tenant_id matches
                if item_tenant_id == tenant_id:
                    try:
                        MondayService.delete_item(item["id"])
                        deleted += 1
                        print(f"✅ Deleted item {item['id']} (tenant: {tenant_id})")
                    except Exception as exc:
                        print(f"⚠️ Failed to delete Monday.com item {item['id']}: {exc}")

            if not cursor:
                break

        return deleted

    @staticmethod
    def count_pending_items_for_tenant_static(
        board_id: str,
        tenant_id: str,
        column_map: Dict[str, str],
        pending_label: str = "Pending",
        batch_size: int = 100,
    ) -> int:
        """
        Count items for a given tenant on a board that are still in 'Pending' status.
        Static method for backward compatibility (uses settings.MONDAY_API_KEY).

        Args:
            board_id: Monday.com board ID
            tenant_id: Tenant ID to filter by (UUID string)
            column_map: Column mapping dictionary (must include "tenant_id" and "status")
            pending_label: The label text used for pending status (default: "Pending")
            batch_size: Number of items to fetch per batch

        Returns:
            Number of items with Status == pending_label for the given tenant.
        """
        tenant_column_id = column_map.get("tenant_id")
        status_column_id = column_map.get("status")
        if not tenant_column_id or not status_column_id:
            raise ValueError("tenant_id or status column not found in board column map")

        pending_count = 0
        cursor: Optional[str] = None

        while True:
            items, cursor = MondayService._fetch_items_with_columns(
                board_id=board_id,
                cursor=cursor,
                limit=batch_size,
                column_ids=[tenant_column_id, status_column_id],
            )

            if not items:
                break

            for item in items:
                item_tenant_id = None
                status_text = None
                for col_val in item.get("column_values", []):
                    col_id = col_val.get("id")
                    if col_id == tenant_column_id:
                        item_tenant_id = (col_val.get("text") or "").strip()
                    elif col_id == status_column_id:
                        status_text = (col_val.get("text") or "").strip()

                if item_tail := item_tenant_id:
                    if item_tail == tenant_id and status_text and status_text.lower() == pending_label.lower():
                        pending_count += 1

            if not cursor:
                break

        return pending_count

    @staticmethod
    def get_items_by_batch_id(
        board_id: str,
        batch_id: str,
        tenant_id: str,
        column_map: Dict[str, str],
        batch_size: int = 100
    ) -> List[Dict]:
        """
        Fetch all items from a board with specific batch_id and tenant_id.
        
        Args:
            board_id: Monday.com board ID
            batch_id: Batch ID to filter by
            tenant_id: Tenant ID to filter by (UUID string)
            column_map: Column mapping dictionary (must include "batch_id" and "tenant_id")
            batch_size: Number of items to fetch per batch
            
        Returns:
            List of items matching the batch_id and tenant_id
        """
        batch_column_id = column_map.get("batch_id")
        tenant_column_id = column_map.get("tenant_id")
        
        if not batch_column_id or not tenant_column_id:
            raise ValueError("batch_id or tenant_id column not found in board column map")
        
        items = []
        cursor: Optional[str] = None
        
        # Also fetch call_session_id column if available
        call_session_column_id = column_map.get("call_session_id")
        column_ids = [batch_column_id, tenant_column_id]
        if call_session_column_id:
            column_ids.append(call_session_column_id)
        
        while True:
            # Fetch items with batch_id, tenant_id, and call_session_id columns
            page_items, cursor = MondayService._fetch_items_with_columns(
                board_id=board_id,
                cursor=cursor,
                limit=batch_size,
                column_ids=column_ids
            )
            
            if not page_items:
                break
            
            for item in page_items:
                # Check if item belongs to this batch and tenant
                item_batch_id = None
                item_tenant_id = None
                
                for col_val in item.get("column_values", []):
                    if col_val.get("id") == batch_column_id:
                        item_batch_id = col_val.get("text", "").strip()
                    elif col_val.get("id") == tenant_column_id:
                        item_tenant_id = col_val.get("text", "").strip()
                
                if item_batch_id == batch_id and item_tenant_id == tenant_id:
                    items.append(item)
            
            if not cursor:
                break
        
        return items

    @staticmethod
    def update_item_call_session_id(
        board_id: str,
        item_id: str,
        call_session_id: str,
        column_map: Dict[str, str]
    ) -> Optional[dict]:
        """
        Update call_session_id column for a Monday.com item.
        
        Args:
            board_id: Monday.com board ID
            item_id: Monday.com item ID
            call_session_id: Call session ID (UUID string)
            column_map: Column mapping dictionary (must include "call_session_id")
            
        Returns:
            Updated item data or None if failed
        """
        call_session_column_id = column_map.get("call_session_id")
        if not call_session_column_id:
            raise ValueError("call_session_id column not found in board column map")
        
        column_values = {call_session_column_id: call_session_id}
        
        query = """
        mutation ($boardId: ID!, $itemId: ID!, $columnValues: JSON!) {
            change_multiple_column_values (
                board_id: $boardId,
                item_id: $itemId,
                column_values: $columnValues
            ) {
                id
            }
        }
        """
        
        variables = {
            "boardId": board_id,
            "itemId": item_id,
            "columnValues": json.dumps(column_values),
        }
        
        try:
            return MondayService._execute_static(query, variables)
        except Exception as exc:
            print(f"⚠️ Failed to update call_session_id for Monday.com item {item_id}: {exc}")
            return None

    @staticmethod
    def update_item_status_and_session_id(
        board_id: str,
        item_id: str,
        status: str,
        call_session_id: Optional[str],
        column_map: Dict[str, str]
    ) -> Optional[dict]:
        """
        Update both status and call_session_id for a Monday.com item in one call.
        
        Args:
            board_id: Monday.com board ID
            item_id: Monday.com item ID
            status: Status to set ("Called" or "Failed")
            call_session_id: Call session ID (UUID string) - optional
            column_map: Column mapping dictionary
            
        Returns:
            Updated item data or None if failed
        """
        status_column_id = column_map.get("status")
        call_session_column_id = column_map.get("call_session_id")
        
        if not status_column_id:
            raise ValueError("status column not found in board column map")
        
        column_values = {
            status_column_id: {"label": status}
        }
        
        if call_session_id and call_session_column_id:
            column_values[call_session_column_id] = call_session_id
        
        query = """
        mutation ($boardId: ID!, $itemId: ID!, $columnValues: JSON!) {
            change_multiple_column_values (
                board_id: $boardId,
                item_id: $itemId,
                column_values: $columnValues
            ) {
                id
            }
        }
        """
        
        variables = {
            "boardId": board_id,
            "itemId": item_id,
            "columnValues": json.dumps(column_values),
        }
        
        try:
            return MondayService._execute_static(query, variables)
        except Exception as exc:
            print(f"⚠️ Failed to update status and call_session_id for Monday.com item {item_id}: {exc}")
            return None

    @staticmethod
    def update_items_email_sent(
        board_id: str,
        item_ids: List[str],
        email_sent_column_id: str
    ) -> int:
        """
        Update Email Sent status to 'Yes' for multiple items.
        
        Args:
            board_id: Monday.com board ID
            item_ids: List of item IDs to update
            email_sent_column_id: Email Sent column ID
            
        Returns:
            Number of items successfully updated
        """
        if not item_ids:
            return 0
        
        # Use label instead of index for status columns (Monday.com API requirement)
        column_values = {email_sent_column_id: {"label": "Yes"}}
        
        # Monday.com API's change_multiple_column_values only accepts item_id (singular)
        # So we need to update each item individually
        updated_count = 0
        
        for item_id in item_ids:
            query = """
            mutation ($boardId: ID!, $itemId: ID!, $columnValues: JSON!) {
                change_multiple_column_values (
                    board_id: $boardId,
                    item_id: $itemId,
                    column_values: $columnValues
                ) {
                    id
                }
            }
            """
            
            variables = {
                "boardId": board_id,
                "itemId": item_id,
                "columnValues": json.dumps(column_values)
            }
            
            try:
                data = MondayService._execute_static(query, variables)
                result = data.get("change_multiple_column_values")
                if result and result.get("id"):
                    updated_count += 1
                    print(f"✅ Updated email sent status for item {item_id}")
                else:
                    print(f"⚠️ No response for item {item_id}, data: {data}")
            except Exception as exc:
                print(f"⚠️ Failed to update email sent status for item {item_id}: {exc}")
                import traceback
                traceback.print_exc()
                continue
        
        return updated_count

    # BaseCRMService implementation methods (instance methods)
    
    def ensure_required_fields(self, container_id: str) -> Dict[str, str]:
        """Ensure required fields (implements BaseCRMService) - uses instance API key from DB"""
        return self._ensure_required_columns_instance(container_id)
    
    def _ensure_required_columns_instance(self, board_id: str) -> Dict[str, str]:
        """
        Ensure the scheduled-calls board has exactly the required columns.
        Uses instance API key (from DB).
        """
        required_titles = {c["title"].lower() for c in MondayService.REQUIRED_COLUMNS}
        columns = self.get_board_columns(board_id)
        type_lookup = {col_def["title"].lower(): col_def["type"] for col_def in MondayService.REQUIRED_COLUMNS}

        # Remove required-title columns with mismatched types to avoid duplicates
        for col in columns:
            title = col["title"].lower()
            if title in required_titles and col.get("type") != type_lookup.get(title):
                try:
                    self.delete_column(board_id, col["id"])
                except Exception as exc:
                    print(f"⚠️ Failed to remove mismatched column {col['id']} on board {board_id}: {exc}")

        # Refresh columns after cleanup
        columns = self.get_board_columns(board_id)
        title_lookup = {col["title"].lower(): col for col in columns}

        # Create missing required columns
        for col_def in MondayService.REQUIRED_COLUMNS:
            current = title_lookup.get(col_def["title"].lower())
            if not current:
                created = self.create_column(
                    board_id=board_id,
                    title=col_def["title"],
                    column_type=col_def["type"],
                    defaults=col_def.get("defaults"),
                )
                title_lookup[col_def["title"].lower()] = created

        # Refresh columns to get final IDs
        columns = self.get_board_columns(board_id)
        title_lookup = {col["title"].lower(): col for col in columns}

        # Remove extraneous columns (keep item name)
        for col in columns:
            title = col["title"].lower()
            if col.get("type") == "name":
                continue
            if title not in required_titles:
                try:
                    self.delete_column(board_id, col["id"])
                except Exception as exc:
                    print(f"⚠️ Failed to delete extra column {col['id']} on board {board_id}: {exc}")

        # Final mapping
        required_map: Dict[str, str] = {}
        for col_def in MondayService.REQUIRED_COLUMNS:
            match = title_lookup.get(col_def["title"].lower())
            if not match:
                raise ValueError(f"Missing required column {col_def['title']} on board {board_id}")
            required_map[col_def["key"]] = match["id"]

        return required_map

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
        """Create scheduled call item (implements BaseCRMService)"""
        # Use instance _execute which uses instance API key
        required_keys = {"status", "agent_id", "call_time_utc", "tenant_id", "user_id"}
        missing = required_keys - set(field_map.keys())
        if missing:
            raise ValueError(f"Missing Monday column ids for: {', '.join(sorted(missing))}")

        column_values = {
            field_map["status"]: {"label": "Pending"},
            field_map["agent_id"]: agent_id,
            field_map["call_time_utc"]: call_time_utc,
            field_map["tenant_id"]: tenant_id,
            field_map["user_id"]: user_id,
        }
        
        if batch_id and "batch_id" in field_map:
            column_values[field_map["batch_id"]] = batch_id
        if phone_number_id and "phone_number_id" in field_map:
            column_values[field_map["phone_number_id"]] = phone_number_id
        if "email_sent" in field_map:
            column_values[field_map["email_sent"]] = {"label": "No"}

        query = """
        mutation ($boardId: ID!, $itemName: String!, $columnValues: JSON!) {
            create_item (
                board_id: $boardId,
                item_name: $itemName,
                column_values: $columnValues
            ) {
                id
                name
            }
        }
        """
        variables = {
            "boardId": container_id,
            "itemName": phone_number,
            "columnValues": json.dumps(column_values),
        }

        try:
            data = self._execute(query, variables)
            return data.get("create_item")
        except Exception as exc:
            print(f"⚠️ Failed to create Monday.com item for {phone_number}: {exc}")
            return None

    def update_item_status(
        self,
        container_id: str,
        item_id: str,
        status: str,
        field_map: Dict[str, str],
    ) -> Optional[dict]:
        """Update item status (implements BaseCRMService)"""
        status_column_id = field_map.get("status", "status")
        column_values = {status_column_id: {"label": status}}

        query = """
        mutation ($boardId: ID!, $itemId: ID!, $columnValues: JSON!) {
            change_multiple_column_values (
                board_id: $boardId,
                item_id: $itemId,
                column_values: $columnValues
            ) {
                id
            }
        }
        """

        variables = {
            "boardId": container_id,
            "itemId": item_id,
            "columnValues": json.dumps(column_values),
        }

        try:
            return self._execute(query, variables)
        except Exception as exc:
            print(f"⚠️ Failed to update Monday.com item {item_id}: {exc}")
            return None

    def update_item_call_session_id(
        self,
        container_id: str,
        item_id: str,
        call_session_id: str,
        field_map: Dict[str, str],
    ) -> Optional[dict]:
        """Update call_session_id (implements BaseCRMService)"""
        call_session_column_id = field_map.get("call_session_id")
        if not call_session_column_id:
            raise ValueError("call_session_id column not found in board column map")
        
        column_values = {call_session_column_id: call_session_id}
        
        query = """
        mutation ($boardId: ID!, $itemId: ID!, $columnValues: JSON!) {
            change_multiple_column_values (
                board_id: $boardId,
                item_id: $itemId,
                column_values: $columnValues
            ) {
                id
            }
        }
        """
        
        variables = {
            "boardId": container_id,
            "itemId": item_id,
            "columnValues": json.dumps(column_values),
        }
        
        try:
            return self._execute(query, variables)
        except Exception as exc:
            print(f"⚠️ Failed to update call_session_id for Monday.com item {item_id}: {exc}")
            return None

    def get_required_fields(self) -> List[Dict]:
        """Get required fields (implements BaseCRMService)"""
        return self.REQUIRED_COLUMNS

    def delete_items_by_tenant(
        self,
        container_id: str,
        tenant_id: str,
        field_map: Dict[str, str],
        batch_size: int = 50
    ) -> int:
        """Delete items by tenant (implements BaseCRMService)"""
        return MondayService.delete_items_by_tenant_static(container_id, tenant_id, field_map, batch_size)

    def _fetch_items_with_columns_instance(self, board_id: str, cursor: Optional[str], limit: int, column_ids: List[str]) -> Tuple[List[Dict], Optional[str]]:
        """Fetch items with specific column values for filtering (instance method - uses instance API key)."""
        query = """
        query ($boardId: [ID!], $cursor: String, $limit: Int!, $columnIds: [String!]) {
            boards (ids: $boardId) {
                items_page (cursor: $cursor, limit: $limit) {
                    cursor
                    items {
                        id
                        name
                        column_values(ids: $columnIds) {
                            id
                            text
                        }
                    }
                }
            }
        }
        """
        data = self._execute(query, {
            "boardId": board_id,
            "cursor": cursor,
            "limit": limit,
            "columnIds": column_ids
        })
        boards = data.get("boards") or []
        if not boards:
            return [], None
        page = boards[0].get("items_page") or {}
        items = page.get("items") or []
        cursor = page.get("cursor")
        return items, cursor

    def count_pending_items_for_tenant(
        self,
        container_id: str,
        tenant_id: str,
        field_map: Dict[str, str],
        pending_label: str = "Pending",
        batch_size: int = 100
    ) -> int:
        """Count pending items by tenant (implements BaseCRMService - uses instance API key)"""
        tenant_column_id = field_map.get("tenant_id")
        status_column_id = field_map.get("status")
        if not tenant_column_id or not status_column_id:
            raise ValueError("tenant_id or status column not found in field map")

        pending_count = 0
        cursor: Optional[str] = None

        while True:
            items, cursor = self._fetch_items_with_columns_instance(
                board_id=container_id,
                cursor=cursor,
                limit=batch_size,
                column_ids=[tenant_column_id, status_column_id],
            )

            if not items:
                break

            for item in items:
                item_tenant_id = None
                status_text = None
                for col_val in item.get("column_values", []):
                    col_id = col_val.get("id")
                    if col_id == tenant_column_id:
                        item_tenant_id = (col_val.get("text") or "").strip()
                    elif col_id == status_column_id:
                        status_text = (col_val.get("text") or "").strip()

                if item_tenant_id == tenant_id and status_text and status_text.lower() == pending_label.lower():
                    pending_count += 1

            if not cursor:
                break

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
        Fetch all items from a board with specific batch_id and tenant_id (instance method).
        Uses instance API key from database.
        
        Args:
            container_id: Monday.com board ID
            batch_id: Batch ID to filter by
            tenant_id: Tenant ID to filter by (UUID string)
            field_map: Field mapping dictionary (must include "batch_id" and "tenant_id")
            batch_size: Number of items to fetch per batch
            
        Returns:
            List of items matching the batch_id and tenant_id
        """
        batch_column_id = field_map.get("batch_id")
        tenant_column_id = field_map.get("tenant_id")
        
        if not batch_column_id or not tenant_column_id:
            raise ValueError("batch_id or tenant_id column not found in board column map")
        
        items = []
        cursor: Optional[str] = None
        
        # Also fetch call_session_id column if available
        call_session_column_id = field_map.get("call_session_id")
        column_ids = [batch_column_id, tenant_column_id]
        if call_session_column_id:
            column_ids.append(call_session_column_id)
        
        while True:
            # Fetch items with batch_id, tenant_id, and call_session_id columns
            page_items, cursor = self._fetch_items_with_columns_instance(
                board_id=container_id,
                cursor=cursor,
                limit=batch_size,
                column_ids=column_ids
            )
            
            if not page_items:
                break
            
            for item in page_items:
                # Check if item belongs to this batch and tenant
                item_batch_id = None
                item_tenant_id = None
                
                for col_val in item.get("column_values", []):
                    if col_val.get("id") == batch_column_id:
                        item_batch_id = col_val.get("text", "").strip()
                    elif col_val.get("id") == tenant_column_id:
                        item_tenant_id = col_val.get("text", "").strip()
                
                if item_batch_id == batch_id and item_tenant_id == tenant_id:
                    items.append(item)
            
            if not cursor:
                break
        
        return items

    def _fetch_items_with_columns_instance(
        self, 
        board_id: str, 
        cursor: Optional[str], 
        limit: int, 
        column_ids: List[str]
    ) -> Tuple[List[Dict], Optional[str]]:
        """Fetch items with specific column values for filtering (instance method)."""
        query = """
        query ($boardId: [ID!], $cursor: String, $limit: Int!, $columnIds: [String!]) {
            boards (ids: $boardId) {
                items_page (cursor: $cursor, limit: $limit) {
                    cursor
                    items {
                        id
                        name
                        column_values(ids: $columnIds) {
                            id
                            text
                        }
                    }
                }
            }
        }
        """
        data = self._execute(query, {
            "boardId": board_id,
            "cursor": cursor,
            "limit": limit,
            "columnIds": column_ids
        })
        boards = data.get("boards") or []
        if not boards:
            return [], None
        page = boards[0].get("items_page") or {}
        items = page.get("items") or []
        next_cursor = page.get("cursor")
        return items, next_cursor

    def update_items_email_sent(
        self,
        container_id: str,
        item_ids: List[str],
        field_map: Dict[str, str],
    ) -> int:
        """
        Update Email Sent status to 'Yes' for multiple items (instance method).
        Uses instance API key from database.
        
        Args:
            container_id: Monday.com board ID
            item_ids: List of item IDs to update
            field_map: Field mapping dictionary (must include "email_sent")
            
        Returns:
            Number of items successfully updated
        """
        if not item_ids:
            return 0
        
        email_sent_column_id = field_map.get("email_sent")
        if not email_sent_column_id:
            print(f"⚠️ Email Sent column ID not found in field map")
            return 0
        
        # Use label instead of index for status columns (Monday.com API requirement)
        column_values = {email_sent_column_id: {"label": "Yes"}}
        
        # Monday.com API's change_multiple_column_values only accepts item_id (singular)
        # So we need to update each item individually
        updated_count = 0
        
        for item_id in item_ids:
            query = """
            mutation ($boardId: ID!, $itemId: ID!, $columnValues: JSON!) {
                change_multiple_column_values (
                    board_id: $boardId,
                    item_id: $itemId,
                    column_values: $columnValues
                ) {
                    id
                }
            }
            """
            
            variables = {
                "boardId": container_id,
                "itemId": item_id,
                "columnValues": json.dumps(column_values)
            }
            
            try:
                data = self._execute(query, variables)
                result = data.get("change_multiple_column_values")
                if result and result.get("id"):
                    updated_count += 1
                    print(f"✅ Updated email sent status for item {item_id}")
                else:
                    print(f"⚠️ No response for item {item_id}, data: {data}")
            except Exception as exc:
                print(f"⚠️ Failed to update email sent status for item {item_id}: {exc}")
                import traceback
                traceback.print_exc()
                continue
        
        return updated_count

    # Note: Instance method _execute is defined above (line 116)
    # Static method _execute_static is for backward compatibility only

    @staticmethod
    def create_board(board_name: str, workspace_id: Optional[str] = None) -> Dict[str, str]:
        """Alias for create_board_static (backward compatibility)"""
        return MondayService.create_board_static(board_name, workspace_id)

