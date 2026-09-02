"""Tests for the shared AWS client factory in app.services.s3_service."""
from unittest.mock import MagicMock, patch

from app.core.config import settings
from app.services import s3_service


def test_static_keys_present_are_included_in_kwargs(monkeypatch):
    monkeypatch.setattr(settings, "AWS_ACCESS_KEY_ID", "AKIAFAKE")
    monkeypatch.setattr(settings, "AWS_SECRET_ACCESS_KEY", "fake-secret")
    monkeypatch.setattr(settings, "AWS_SESSION_TOKEN", "")
    monkeypatch.setattr(settings, "AWS_REGION_NAME", "us-east-1")

    kwargs = s3_service._client_kwargs()

    assert kwargs["aws_access_key_id"] == "AKIAFAKE"
    assert kwargs["aws_secret_access_key"] == "fake-secret"
    assert kwargs["region_name"] == "us-east-1"
    assert "aws_session_token" not in kwargs


def test_empty_keys_are_omitted_for_iam_task_role_fallback(monkeypatch):
    monkeypatch.setattr(settings, "AWS_ACCESS_KEY_ID", "")
    monkeypatch.setattr(settings, "AWS_SECRET_ACCESS_KEY", "")
    monkeypatch.setattr(settings, "AWS_SESSION_TOKEN", "")
    monkeypatch.setattr(settings, "AWS_REGION_NAME", "us-east-1")

    kwargs = s3_service._client_kwargs()

    assert "aws_access_key_id" not in kwargs
    assert "aws_secret_access_key" not in kwargs
    assert "aws_session_token" not in kwargs
    assert kwargs["region_name"] == "us-east-1"


def test_session_token_included_alongside_static_keys(monkeypatch):
    monkeypatch.setattr(settings, "AWS_ACCESS_KEY_ID", "AKIAFAKE")
    monkeypatch.setattr(settings, "AWS_SECRET_ACCESS_KEY", "fake-secret")
    monkeypatch.setattr(settings, "AWS_SESSION_TOKEN", "fake-session-token")
    monkeypatch.setattr(settings, "AWS_REGION_NAME", "us-east-1")

    kwargs = s3_service._client_kwargs()

    assert kwargs["aws_session_token"] == "fake-session-token"
    assert kwargs["aws_access_key_id"] == "AKIAFAKE"
    assert kwargs["aws_secret_access_key"] == "fake-secret"


def test_session_token_without_static_keys_is_dropped(monkeypatch):
    monkeypatch.setattr(settings, "AWS_ACCESS_KEY_ID", "")
    monkeypatch.setattr(settings, "AWS_SECRET_ACCESS_KEY", "")
    monkeypatch.setattr(settings, "AWS_SESSION_TOKEN", "fake-session-token")
    monkeypatch.setattr(settings, "AWS_REGION_NAME", "us-east-1")

    kwargs = s3_service._client_kwargs()

    assert "aws_session_token" not in kwargs
    assert "aws_access_key_id" not in kwargs
    assert "aws_secret_access_key" not in kwargs


def test_verify_aws_caller_identity_returns_identity_on_success(monkeypatch):
    monkeypatch.setattr(settings, "AWS_ACCESS_KEY_ID", "AKIAFAKE")
    monkeypatch.setattr(settings, "AWS_SECRET_ACCESS_KEY", "fake-secret")
    monkeypatch.setattr(settings, "AWS_SESSION_TOKEN", "")
    monkeypatch.setattr(settings, "AWS_REGION_NAME", "us-east-1")

    fake_identity = {
        "Arn": "arn:aws:iam::123456789012:role/fake-role",
        "Account": "123456789012",
        "UserId": "AROAFAKE",
    }
    mock_sts_client = MagicMock()
    mock_sts_client.get_caller_identity.return_value = fake_identity

    with patch("app.services.s3_service.boto3.client", return_value=mock_sts_client) as mock_client:
        result = s3_service.verify_aws_caller_identity()

    mock_client.assert_called_once()
    assert mock_client.call_args[0][0] == "sts"
    assert result == fake_identity


def test_verify_aws_caller_identity_returns_none_on_failure(monkeypatch):
    monkeypatch.setattr(settings, "AWS_ACCESS_KEY_ID", "")
    monkeypatch.setattr(settings, "AWS_SECRET_ACCESS_KEY", "")
    monkeypatch.setattr(settings, "AWS_SESSION_TOKEN", "")
    monkeypatch.setattr(settings, "AWS_REGION_NAME", "us-east-1")

    mock_sts_client = MagicMock()
    mock_sts_client.get_caller_identity.side_effect = RuntimeError("no credentials found")

    with patch("app.services.s3_service.boto3.client", return_value=mock_sts_client):
        result = s3_service.verify_aws_caller_identity()

    assert result is None
