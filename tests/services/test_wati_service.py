"""WATI API client — HTTP mocked."""

from unittest.mock import MagicMock, patch

import pytest

from app.core.config import settings


@pytest.fixture
def wati_env(monkeypatch):
    monkeypatch.setattr(settings, "WATI_ENABLED", True)
    monkeypatch.setattr(settings, "WATI_API_BASE_URL", "https://live.example.wati.io")
    monkeypatch.setattr(settings, "WATI_ACCESS_TOKEN", "test-bearer-token")
    monkeypatch.setattr(settings, "WATI_TEMPLATE_NAME", "booking_staff_prompt")
    monkeypatch.setattr(settings, "WATI_CHANNEL_NUMBER", "15550001001")


def test_send_template_message_posts_json(wati_env):
    from app.services.wati_service import wati_service

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"result": True, "phone_number": "923001234567"}

    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = lambda *a: None

    with patch("app.services.wati_service.httpx.Client", return_value=mock_client):
        out = wati_service.send_template_message(
            whatsapp_number="+923001234567",
            template_name="booking_staff_prompt",
            parameters=[{"name": "a", "value": "b"}],
            broadcast_name="test_broadcast",
        )
    assert out["result"] is True
    mock_client.post.assert_called_once()
    call_kw = mock_client.post.call_args
    assert "sendTemplateMessage" in call_kw[0][0]
    assert call_kw[1]["params"]["whatsappNumber"] == "923001234567"
