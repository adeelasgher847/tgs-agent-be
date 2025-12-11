"""
Monday.com API Service for Scheduled Calls Integration
"""

import json
from typing import Dict, List, Optional, Tuple

import requests

from app.core.config import settings


class MondayService:
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
        {
            "key": "email_sent",
            "title": "Email Sent",
            "type": "status",
            "defaults": {"labels": {"0": "No", "1": "Yes"}},
        },
    ]

    @staticmethod
    def build_board_url(board_id: str) -> str:
        return f"https://app.monday.com/boards/{board_id}"

    @staticmethod
    def _headers() -> Dict[str, str]:
        if not settings.MONDAY_API_KEY:
            raise ValueError("Monday.com API key is not configured")
        return {
            "Authorization": settings.MONDAY_API_KEY,
            "Content-Type": "application/json",
            "API-Version": "2024-01",
        }

    @staticmethod
    def _execute(query: str, variables: Dict) -> Dict:
        response = requests.post(
            MondayService.API_URL,
            json={"query": query, "variables": variables},
            headers=MondayService._headers(),
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        if "errors" in payload:
            raise ValueError(payload["errors"])
        return payload.get("data", {})

    @staticmethod
    def create_board(board_name: str, workspace_id: Optional[str] = None) -> Dict[str, str]:
        """
        Create a dedicated Monday.com board for a tenant.
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

        data = MondayService._execute(query, variables)
        board = data.get("create_board")
        if not board:
            raise ValueError("Failed to create Monday.com board")

        board_id = str(board["id"])
        return {"id": board_id, "url": MondayService.build_board_url(board_id)}

    @staticmethod
    def get_board_columns(board_id: str) -> List[Dict]:
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
        data = MondayService._execute(query, {"boardId": board_id})
        boards = data.get("boards") or []
        if not boards:
            raise ValueError(f"Board {board_id} not found")
        return boards[0].get("columns", [])

    @staticmethod
    def create_column(board_id: str, title: str, column_type: str, defaults: Optional[Dict] = None) -> Dict:
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
        data = MondayService._execute(query, variables)
        column = data.get("create_column")
        if not column:
            raise ValueError(f"Failed to create column {title} on board {board_id}")
        return column

    @staticmethod
    def delete_column(board_id: str, column_id: str) -> None:
        query = """
        mutation ($boardId: ID!, $columnId: String!) {
            delete_column (board_id: $boardId, column_id: $columnId) {
                id
            }
        }
        """
        MondayService._execute(query, {"boardId": board_id, "columnId": column_id})

    @staticmethod
    def ensure_required_columns(board_id: str) -> Dict[str, str]:
        """
        Ensure the scheduled-calls board has exactly the required columns.

        Returns:
            Dict mapping required column keys to their Monday column IDs.
        """
        required_titles = {c["title"].lower() for c in MondayService.REQUIRED_COLUMNS}
        columns = MondayService.get_board_columns(board_id)
        type_lookup = {col_def["title"].lower(): col_def["type"] for col_def in MondayService.REQUIRED_COLUMNS}

        # Remove required-title columns with mismatched types to avoid duplicates
        for col in columns:
            title = col["title"].lower()
            if title in required_titles and col.get("type") != type_lookup.get(title):
                try:
                    MondayService.delete_column(board_id, col["id"])
                except Exception as exc:  # pragma: no cover - defensive logging
                    print(f"⚠️ Failed to remove mismatched column {col['id']} on board {board_id}: {exc}")

        # Refresh columns after cleanup
        columns = MondayService.get_board_columns(board_id)
        title_lookup = {col["title"].lower(): col for col in columns}

        # Create missing required columns
        for col_def in MondayService.REQUIRED_COLUMNS:
            current = title_lookup.get(col_def["title"].lower())
            if not current:
                created = MondayService.create_column(
                    board_id=board_id,
                    title=col_def["title"],
                    column_type=col_def["type"],
                    defaults=col_def.get("defaults"),
                )
                title_lookup[col_def["title"].lower()] = created

        # Refresh columns to get final IDs
        columns = MondayService.get_board_columns(board_id)
        title_lookup = {col["title"].lower(): col for col in columns}

        # Remove extraneous columns (keep item name)
        for col in columns:
            title = col["title"].lower()
            if col.get("type") == "name":
                continue
            if title not in required_titles:
                try:
                    MondayService.delete_column(board_id, col["id"])
                except Exception as exc:  # pragma: no cover - defensive logging
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
            data = MondayService._execute(query, variables)
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
            return MondayService._execute(query, variables)
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
        MondayService._execute(query, {"itemId": item_id})

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
        data = MondayService._execute(query, {
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

    @staticmethod
    def delete_items_by_tenant(board_id: str, tenant_id: str, column_map: Dict[str, str], batch_size: int = 50) -> int:
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
            return MondayService._execute(query, variables)
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
            return MondayService._execute(query, variables)
        except Exception as exc:
            print(f"⚠️ Failed to update status and call_session_id for Monday.com item {item_id}: {exc}")
            return None

