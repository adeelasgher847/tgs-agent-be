
import pytest
from unittest.mock import MagicMock, patch
from app.services.trello_service import TrelloService

class TestTrelloService:
    
    @patch('requests.get')
    def test_get_items_by_batch_id_fetches_status_from_custom_fields(self, mock_get):
        """
        Test that get_items_by_batch_id extracts status from custom fields.
        """
        service = TrelloService("api_key", "api_token")
        
        # Setup mock data
        container_id = "board123"
        batch_id = "batch_123"
        tenant_id = "tenant_456"
        
        field_map = {
            "batch_id": "field_batch",
            "tenant_id": "field_tenant",
            "status": "field_status"
        }
        
        # Mock cards response
        cards = [{
            "id": "card1",
            "name": "Card 1",
            "desc": ""
        }]
        
        # Mock custom fields response
        custom_fields = [
            {"id": "field_batch", "value": {"text": batch_id}},
            {"id": "field_tenant", "value": {"text": tenant_id}},
            {"id": "field_status", "value": {"text": "Called"}}
        ]
        
        # Configure mock side effects
        def side_effect(url, params=None, timeout=None):
            if "customFields" in url:
                # Custom fields request (contains /card/ or /cards/)
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = custom_fields
                return mock_resp
            elif "/cards" in url:
                # Board cards request
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = cards
                return mock_resp
            return MagicMock()

        mock_get.side_effect = side_effect
        
        # Execute
        items = service.get_items_by_batch_id(
            container_id=container_id,
            batch_id=batch_id,
            tenant_id=tenant_id,
            field_map=field_map
        )
        
        # Verify
        assert len(items) == 1
        item = items[0]
        
        # Check column values for status
        status_val = next((fv for fv in item["column_values"] if fv["id"] == "field_status"), None)
        assert status_val is not None
        assert status_val["text"] == "Called"

    @patch('requests.get')
    def test_get_items_by_batch_id_fetches_status_from_description(self, mock_get):
        """
        Test fallback to description parsing if custom fields missing.
        """
        service = TrelloService("api_key", "api_token")
        
        # Setup mock data
        container_id = "board123"
        batch_id = "batch_123"
        tenant_id = "tenant_456"
        
        field_map = {
            "batch_id": "field_batch",
            "tenant_id": "field_tenant",
            "status": "field_status"
        }
        
        # Mock cards response with description
        cards = [{
            "id": "card1",
            "name": "Card 1",
            "desc": f"Batch ID: {batch_id}\nTenant ID: {tenant_id}\nStatus: Failed"
        }]
        
        # Mock empty custom fields response (simulating no Power-Ups)
        custom_fields = []
        
        # Configure mock side_effect
        def side_effect(url, params=None, timeout=None):
            if "customFields" in url:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = custom_fields
                return mock_resp
            elif "/cards" in url:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = cards
                return mock_resp
            return MagicMock()

        mock_get.side_effect = side_effect
        
        # Execute
        items = service.get_items_by_batch_id(
            container_id=container_id,
            batch_id=batch_id,
            tenant_id=tenant_id,
            field_map=field_map
        )
        
        # Verify
        assert len(items) == 1
        item = items[0]
        
        # Check column values for status
        status_val = next((fv for fv in item["column_values"] if fv["id"] == "field_status"), None)
        assert status_val is not None
        assert status_val["text"] == "Failed"
