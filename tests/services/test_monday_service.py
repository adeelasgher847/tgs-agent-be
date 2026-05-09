
import pytest
from unittest.mock import MagicMock, patch
from app.services.monday_service import MondayService

class TestMondayService:
    
    @patch('app.services.monday_service.MondayService._fetch_items_with_columns_instance')
    def test_get_items_by_batch_id_fetches_status_column(self, mock_fetch):
        """
        Test that get_items_by_batch_id includes the 'status' column in the column_ids list.
        This verifies the fix for batch analysis statistics.
        """
        # Setup
        service = MondayService("fake_api_key")
        container_id = "12345"
        batch_id = "batch_123"
        tenant_id = "tenant_456"
        
        # Field map with all required fields
        field_map = {
            "batch_id": "col_batch",
            "tenant_id": "col_tenant",
            "call_session_id": "col_session",
            "status": "col_status"  # <--- CRITICAL: This must be in the fetch list
        }
        
        # Mock response from fetch
        mock_fetch.return_value = ([], None)
        
        # Execute
        service.get_items_by_batch_id(
            container_id=container_id,
            batch_id=batch_id,
            tenant_id=tenant_id,
            field_map=field_map
        )
        
        # Verify
        # Check the arguments passed to _fetch_items_with_columns_instance
        call_args = mock_fetch.call_args
        _, kwargs = call_args
        column_ids_passed = kwargs.get('column_ids', [])
        
        # Assertions
        assert "col_batch" in column_ids_passed, "batch_id column should be fetched"
        assert "col_tenant" in column_ids_passed, "tenant_id column should be fetched"
        assert "col_session" in column_ids_passed, "call_session_id column should be fetched"
        assert "col_status" in column_ids_passed, "status column should be fetched (FIX VERIFICATION)"
        
    @patch('app.services.monday_service.MondayService._fetch_items_with_columns_instance')
    def test_get_items_by_batch_id_filters_correctly(self, mock_fetch):
        """
        Test that it correctly filters items by batch_id and tenant_id
        """
        service = MondayService("fake_api_key")
        
        # Mock items return
        mock_items = [
            {
                "id": "item1", 
                "column_values": [
                    {"id": "col_batch", "text": "batch_123"},
                    {"id": "col_tenant", "text": "tenant_456"},
                    {"id": "col_status", "text": "Called"}
                ]
            },
            {
                "id": "item2", 
                "column_values": [
                    {"id": "col_batch", "text": "batch_WRONG"}, # Should be filtered out
                    {"id": "col_tenant", "text": "tenant_456"},
                    {"id": "col_status", "text": "Pending"}
                ]
            }
        ]
        
        mock_fetch.return_value = (mock_items, None)
        
        field_map = {
            "batch_id": "col_batch",
            "tenant_id": "col_tenant",
            "status": "col_status"
        }
        
        # Execute
        result = service.get_items_by_batch_id(
            container_id="123",
            batch_id="batch_123",
            tenant_id="tenant_456",
            field_map=field_map
        )
        
        # Verify
        assert len(result) == 1
        assert result[0]["id"] == "item1"
