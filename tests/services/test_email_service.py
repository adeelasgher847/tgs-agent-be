from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from app.services.email_service import EmailService, _html_to_text


@pytest.fixture
def email_service_instance():
    with patch("app.services.email_service.get_ses_client") as mock_get_ses_client:
        mock_ses = MagicMock()
        mock_get_ses_client.return_value = mock_ses
        service = EmailService()
        service.sender_email = "sender@example.com"
        yield service, mock_ses


def _success_response():
    return {"MessageId": "abc123", "ResponseMetadata": {"HTTPStatusCode": 200}}


def _client_error(code="MessageRejected", message="Email address is not verified."):
    return ClientError(
        error_response={"Error": {"Code": code, "Message": message}},
        operation_name="SendEmail",
    )


class TestHtmlToText:
    def test_strips_tags(self):
        html = "<html><body><h2>Hi</h2><p>Hello <strong>World</strong></p></body></html>"
        text = _html_to_text(html)
        assert "<" not in text
        assert "Hello" in text and "World" in text


class TestSendEmailSuccess:
    def test_sends_html_email(self, email_service_instance):
        service, mock_ses = email_service_instance
        mock_ses.send_email.return_value = _success_response()

        result = service._send_email(
            to_email="user@example.com",
            subject="Test Subject",
            html_body="<p>Hello</p>",
        )

        assert result is True
        mock_ses.send_email.assert_called_once()
        kwargs = mock_ses.send_email.call_args.kwargs
        assert kwargs["Source"] == "sender@example.com"
        assert kwargs["Destination"] == {"ToAddresses": ["user@example.com"]}
        assert kwargs["Message"]["Subject"]["Data"] == "Test Subject"
        assert kwargs["Message"]["Body"]["Html"]["Data"] == "<p>Hello</p>"

    def test_generates_plain_text_fallback_when_missing(self, email_service_instance):
        service, mock_ses = email_service_instance
        mock_ses.send_email.return_value = _success_response()

        service._send_email(
            to_email="user@example.com",
            subject="Test",
            html_body="<p>Hello <strong>World</strong></p>",
        )

        body = mock_ses.send_email.call_args.kwargs["Message"]["Body"]
        assert "Text" in body
        assert "Hello" in body["Text"]["Data"]
        assert "<" not in body["Text"]["Data"]

    def test_uses_provided_text_body(self, email_service_instance):
        service, mock_ses = email_service_instance
        mock_ses.send_email.return_value = _success_response()

        service._send_email(
            to_email="user@example.com",
            subject="Test",
            html_body="<p>Hello</p>",
            text_body="Plain text version",
        )

        body = mock_ses.send_email.call_args.kwargs["Message"]["Body"]
        assert body["Text"]["Data"] == "Plain text version"


class TestSendEmailCc:
    def test_includes_valid_cc_recipients(self, email_service_instance):
        service, mock_ses = email_service_instance
        mock_ses.send_email.return_value = _success_response()

        service._send_email(
            to_email="user@example.com",
            subject="Test",
            html_body="<p>Hello</p>",
            cc_emails=["cc1@example.com", "cc2@example.com"],
        )

        destination = mock_ses.send_email.call_args.kwargs["Destination"]
        assert destination["CcAddresses"] == ["cc1@example.com", "cc2@example.com"]

    def test_filters_out_empty_cc_entries(self, email_service_instance):
        service, mock_ses = email_service_instance
        mock_ses.send_email.return_value = _success_response()

        service._send_email(
            to_email="user@example.com",
            subject="Test",
            html_body="<p>Hello</p>",
            cc_emails=["", None, "  ", "valid@example.com"],
        )

        destination = mock_ses.send_email.call_args.kwargs["Destination"]
        assert destination["CcAddresses"] == ["valid@example.com"]

    def test_no_cc_key_when_no_cc_emails(self, email_service_instance):
        service, mock_ses = email_service_instance
        mock_ses.send_email.return_value = _success_response()

        service._send_email(
            to_email="user@example.com",
            subject="Test",
            html_body="<p>Hello</p>",
        )

        destination = mock_ses.send_email.call_args.kwargs["Destination"]
        assert "CcAddresses" not in destination


class TestSendEmailErrors:
    def test_missing_sender_email_returns_false(self, email_service_instance):
        service, mock_ses = email_service_instance
        service.sender_email = ""

        result = service._send_email(
            to_email="user@example.com", subject="Test", html_body="<p>Hi</p>"
        )

        assert result is False
        mock_ses.send_email.assert_not_called()

    @pytest.mark.parametrize(
        "error_code",
        [
            "MessageRejected",
            "MailFromDomainNotVerifiedException",
            "AccountSendingPausedException",
            "LimitExceededException",
        ],
    )
    def test_client_error_returns_false(self, email_service_instance, error_code):
        service, mock_ses = email_service_instance
        mock_ses.send_email.side_effect = _client_error(code=error_code)

        result = service._send_email(
            to_email="user@example.com", subject="Test", html_body="<p>Hi</p>"
        )

        assert result is False

    def test_unexpected_exception_returns_false(self, email_service_instance):
        service, mock_ses = email_service_instance
        mock_ses.send_email.side_effect = RuntimeError("boom")

        result = service._send_email(
            to_email="user@example.com", subject="Test", html_body="<p>Hi</p>"
        )

        assert result is False


class TestStagingMode:
    def test_staging_short_circuits_without_calling_ses(self, email_service_instance):
        service, mock_ses = email_service_instance
        with patch("app.services.email_service.settings.ENVIRONMENT", "staging"):
            result = service._send_email(
                to_email="user@example.com", subject="Test", html_body="<p>Hi</p>"
            )

        assert result is True
        mock_ses.send_email.assert_not_called()


class TestPublicMethods:
    def test_send_invite_email(self, email_service_instance):
        service, mock_ses = email_service_instance
        mock_ses.send_email.return_value = _success_response()

        result = service.send_invite_email(
            "invitee@example.com", "token123", "Alice", "Acme Corp"
        )

        assert result is True
        mock_ses.send_email.assert_called_once()

    def test_send_password_reset_email(self, email_service_instance):
        service, mock_ses = email_service_instance
        mock_ses.send_email.return_value = _success_response()

        result = service.send_password_reset_email(
            "user@example.com", "reset-token", "Bob"
        )

        assert result is True
        mock_ses.send_email.assert_called_once()

    def test_send_data_export_ready_email(self, email_service_instance):
        service, mock_ses = email_service_instance
        mock_ses.send_email.return_value = _success_response()

        result = service.send_data_export_ready_email(
            "admin@example.com", "https://example.com/export.zip", "Acme Corp"
        )

        assert result is True
        mock_ses.send_email.assert_called_once()

    def test_send_generic_email_with_cc(self, email_service_instance):
        service, mock_ses = email_service_instance
        mock_ses.send_email.return_value = _success_response()

        result = service.send_generic_email(
            "user@example.com",
            "Subject",
            "<p>Body</p>",
            cc_emails=["cc@example.com"],
        )

        assert result is True
        destination = mock_ses.send_email.call_args.kwargs["Destination"]
        assert destination["CcAddresses"] == ["cc@example.com"]
