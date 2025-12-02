"""
Monday.com API Service for Scheduled Calls Integration
"""

import requests
import json
from typing import Optional
from app.core.config import settings


class MondayService:
    """Service for interacting with Monday.com API"""
    
    API_URL = "https://api.monday.com/v2"
    
    @staticmethod
    def create_scheduled_call_item(
        schedule_id: str,
        phone_number: str,
        agent_id: str,
        call_time_utc: str,
        tenant_id: str,
        user_id: str
    ) -> Optional[dict]:
        """
        Create a scheduled call item in Monday.com board.
        
        Args:
            schedule_id: Unique UUID for this scheduled call
            phone_number: Phone number to call (e.g., +1234567890)
            agent_id: Agent UUID
            call_time_utc: ISO format datetime string (e.g., 2024-12-02T15:30:00Z)
            tenant_id: Tenant UUID
            user_id: User UUID
        
        Returns:
            dict: Monday.com API response with item_id, or None if failed
        """
        if not settings.MONDAY_API_KEY or not settings.MONDAY_BOARD_ID:
            print("⚠️ Monday.com not configured (MONDAY_API_KEY or MONDAY_BOARD_ID missing)")
            return None
        
        # GraphQL mutation to create item
        query = '''
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
        '''
        
        # Column values - these need to match your Monday.com board column IDs
        # Note: You may need to adjust column IDs based on your actual Monday.com board setup
        column_values = {
            "text": phone_number,  # Simple text columns
            "text4": agent_id,
            "text7": call_time_utc,
            "text8": tenant_id,
            "text9": user_id,
            "text5": schedule_id,
            "status": {"label": "Pending"}
        }
        
        variables = {
            "boardId": settings.MONDAY_BOARD_ID,
            "itemName": f"Call to {phone_number}",
            "columnValues": json.dumps(column_values)
        }
        
        headers = {
            "Authorization": settings.MONDAY_API_KEY,
            "Content-Type": "application/json",
            "API-Version": "2024-01"
        }
        
        try:
            response = requests.post(
                MondayService.API_URL,
                json={"query": query, "variables": variables},
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            if "data" in result and "create_item" in result["data"]:
                item_id = result["data"]["create_item"]["id"]
                print(f"✅ Monday.com item created (ID: {item_id}) for schedule_id: {schedule_id}")
                return result["data"]["create_item"]
            else:
                error_msg = result.get("errors", result)
                print(f"⚠️ Monday.com API error: {error_msg}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Failed to create Monday.com item: {e}")
            return None
        except Exception as e:
            print(f"⚠️ Unexpected error creating Monday.com item: {e}")
            return None
    
    @staticmethod
    def update_item_status(
        item_id: str,
        status: str,
        call_sid: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> Optional[dict]:
        """
        Update Monday.com item status after call attempt.
        
        Args:
            item_id: Monday.com item ID
            status: Status label (e.g., "Called", "Failed", "Waiting")
            call_sid: Twilio call SID (optional)
            error_message: Error message if failed (optional)
        
        Returns:
            dict: Monday.com API response, or None if failed
        """
        if not settings.MONDAY_API_KEY or not settings.MONDAY_BOARD_ID:
            return None
        
        query = '''
        mutation ($boardId: ID!, $itemId: ID!, $columnValues: JSON!) {
            change_multiple_column_values (
                board_id: $boardId,
                item_id: $itemId,
                column_values: $columnValues
            ) {
                id
            }
        }
        '''
        
        column_values = {"status": {"label": status}}
        
        if call_sid:
            column_values["text6"] = call_sid  # call_sid column
        
        if error_message:
            column_values["long_text"] = error_message  # error_message column
        
        variables = {
            "boardId": settings.MONDAY_BOARD_ID,
            "itemId": item_id,
            "columnValues": json.dumps(column_values)
        }
        
        headers = {
            "Authorization": settings.MONDAY_API_KEY,
            "Content-Type": "application/json",
            "API-Version": "2024-01"
        }
        
        try:
            response = requests.post(
                MondayService.API_URL,
                json={"query": query, "variables": variables},
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            print(f"✅ Monday.com item {item_id} updated to status: {status}")
            return response.json()
        except Exception as e:
            print(f"⚠️ Failed to update Monday.com item: {e}")
            return None

