
import pytest
from unittest.mock import MagicMock, patch
from app.services.clickup_service import ClickUpService

class TestClickUpService:
    
    @patch('app.services.clickup_service.ClickUpService._headers')
    @patch('requests.get')
    def test_get_items_by_batch_id_resolves_dropdown_status_label(self, mock_get, mock_headers):
        """
        Test that get_items_by_batch_id resolves status orderindex/UUID to label.
        """
        mock_headers.return_value = {"Authorization": "Bearer key"}
        service = ClickUpService("fake_api_key")
        
        # Setup mock data
        container_id = "list123"
        batch_id = "batch_123"
        tenant_id = "tenant_456"
        
        field_map = {
            "batch_id": "field_batch",
            "tenant_id": "field_tenant",
            "status": "field_status"
        }
        
        # Status field definition with type_config
        status_field_def = {
            "id": "field_status", 
            "value": 1, # Index 1 = "Called"
            "type": "drop_down",
            "type_config": {
                "options": [
                    {"id": "opt1", "name": "Pending", "orderindex": 0},
                    {"id": "opt2", "name": "Called", "orderindex": 1, "label": "Called"}, # Matches value 1
                    {"id": "opt3", "name": "Failed", "orderindex": 2}
                ]
            }
        }
        
        # Mock task response
        tasks = [{
            "id": "task1",
            "name": "Task 1",
            "custom_fields": [
                {"id": "field_batch", "value": batch_id},
                {"id": "field_tenant", "value": tenant_id},
                status_field_def
            ]
        }]
        
        # Configure mock response
        def side_effect(url, headers=None, params=None, timeout=None):
            if "/task/task1" in url and "/list/" not in url:
                # Task detail request - return task object with custom fields
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                # Task detail returns the task object directly
                mock_resp.json.return_value = {
                    "id": "task1", 
                    "name": "Task 1",
                    "custom_fields": [
                        {"id": "field_batch", "value": batch_id},
                        {"id": "field_tenant", "value": tenant_id},
                        status_field_def
                    ]
                }
                return mock_resp
            else:
                # List request
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"tasks": tasks}
                return mock_resp

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
        # Should be resolved to "Called"
        assert status_val["text"] == "Called"
        # Value is also updated to label in current implementation
        assert status_val["value"] == "Called"

    @patch('app.services.clickup_service.ClickUpService._headers')
    @patch('requests.get')
    def test_get_items_by_batch_id_fetches_task_details_if_needed(self, mock_get, mock_headers):
        """
        Test that it fetches individual task details if custom fields are missing/incomplete in list.
        """
        mock_headers.return_value = {"Authorization": "Bearer key"}
        service = ClickUpService("fake_api_key")
        
        # Setup mock data
        container_id = "list123"
        batch_id = "batch_123"
        tenant_id = "tenant_456"
        
        field_map = {
            "batch_id": "field_batch",
            "tenant_id": "field_tenant"
        }
        
        # Mock list response (incomplete fields)
        list_tasks = [{
            "id": "task1",
            "name": "Task 1",
            "custom_fields": [] # Empty in list view
        }]
        
        # Mock task detail response (complete fields)
        task_detail = {
            "id": "task1",
            "name": "Task 1",
            "custom_fields": [
                {"id": "field_batch", "value": batch_id},
                {"id": "field_tenant", "value": tenant_id}
            ]
        }
        
        # Configure mock side effects
        def side_effect(url, headers=None, params=None, timeout=None):
            if "/task/task1" in url and "/list/" not in url:
                # Task detail request
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = task_detail
                return mock_resp
            else:
                # List request
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"tasks": list_tasks}
                return mock_resp

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
        assert items[0]["id"] == "task1"
