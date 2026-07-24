import logging
import re
from typing import List, Optional

from botocore.exceptions import ClientError

from app.core.config import settings
from app.services.s3_service import get_ses_client

logger = logging.getLogger(__name__)

# SES error codes that indicate a sandbox/verification restriction rather
# than a transient failure — logged distinctly so they're easy to spot.
_SANDBOX_ERROR_CODES = {
    "MessageRejected",
    "MailFromDomainNotVerifiedException",
    "AccountSendingPausedException",
    "LimitExceededException",
}


def build_invite_email_content(
    invite_token: str, inviter_name: str, tenant_name: str
) -> tuple[str, str]:
    """Subject + HTML body for workspace invite."""
    subject = f"You're invited to join {tenant_name} on Voice Agent Platform"
    invite_link = f"{settings.FRONTEND_URL}/accept-invite?token={invite_token}"
    html_body = f"""
            <html>
            <body>
                <h2>You're Invited to Join {tenant_name}!</h2>
                <p>Hello,</p>
                <p>{inviter_name} has invited you to join the <strong>{tenant_name}</strong> team on Voice Agent Platform.</p>
                <p>Click the link below to accept the invitation and create your account:</p>
                <p><a href="{invite_link}" style="background-color: #28a745; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block;">Accept Invitation</a></p>
                <p>Or copy and paste this link into your browser:</p>
                <p>{invite_link}</p>
                <p><strong>This invitation will expire in 7 days.</strong></p>
                <p>If you don't want to join this team, you can safely ignore this email.</p>
                <p>Best regards,<br>The Voice Agent Team</p>
            </body>
            </html>
            """
    return subject, html_body


def _html_to_text(html_body: str) -> str:
    """Best-effort plain-text fallback derived from an HTML body."""
    text = re.sub(r"<[^>]+>", "", html_body)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


class EmailService:
    def __init__(self):
        self.sender_email = settings.AWS_SES_SENDER_EMAIL
        self.ses_client = get_ses_client()

    def send_password_reset_email(self, email: str, reset_token: str, user_name: str) -> bool:
        """
        Send password reset email to user

        Args:
            email: User's email address
            reset_token: Password reset token
            user_name: User's name for personalization

        Returns:
            bool: True if email sent successfully, False otherwise
        """
        try:
            subject = "Password Reset Request - Voice Agent Platform"
            reset_link = f"{settings.FRONTEND_URL}/reset-password?token={reset_token}"
            body = f"""
            <html>
            <body>
                <h2>Password Reset Request</h2>
                <p>Hello {user_name},</p>
                <p>We received a request to reset your password for your Voice Agent Platform account.</p>
                <p>Click the link below to reset your password:</p>
                <p><a href="{reset_link}" style="background-color: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Reset Password</a></p>
                <p>Or copy and paste this link into your browser:</p>
                <p>{reset_link}</p>
                <p><strong>This link will expire in {settings.PASSWORD_RESET_TOKEN_EXPIRE_MINUTES} minutes.</strong></p>
                <p>If you didn't request this password reset, please ignore this email. Your password will remain unchanged.</p>
                <p>Best regards,<br>The Voice Agent Team</p>
            </body>
            </html>
            """
            return self._send_email(to_email=email, subject=subject, html_body=body)
        except Exception as e:
            logger.error(f"Error sending password reset email to {email}: {str(e)}")
            return False

    def _send_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        cc_emails: Optional[List[str]] = None,
        text_body: Optional[str] = None,
    ) -> bool:
        """
        Send an email using AWS SES.

        In staging, emails are logged to the console instead of actually
        sent — avoids real mail going out from a non-production environment.
        """
        if settings.ENVIRONMENT == "staging":
            logger.info(
                "[STAGING EMAIL] to=%s cc=%s subject=%s\n%s",
                to_email, cc_emails or [], subject, html_body,
            )
            return True

        if not self.sender_email:
            logger.error("AWS SES sender is not configured. Missing AWS_SES_SENDER_EMAIL.")
            return False

        if not text_body:
            text_body = _html_to_text(html_body)

        destination = {"ToAddresses": [to_email]}
        if cc_emails:
            valid_ccs = [cc for cc in cc_emails if cc and cc.strip()]
            if valid_ccs:
                destination["CcAddresses"] = valid_ccs

        body_dict = {
            "Html": {"Data": html_body, "Charset": "UTF-8"},
            "Text": {"Data": text_body, "Charset": "UTF-8"},
        }

        try:
            response = self.ses_client.send_email(
                Source=self.sender_email,
                Destination=destination,
                Message={
                    "Subject": {"Data": subject, "Charset": "UTF-8"},
                    "Body": body_dict,
                },
            )
            status_code = response.get("ResponseMetadata", {}).get("HTTPStatusCode", 200)
            if 200 <= status_code < 300:
                logger.info(f"Email sent successfully to {to_email}")
                return True
            logger.error(f"Failed to send email to {to_email}. Status: {status_code}")
            return False
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            if error_code in _SANDBOX_ERROR_CODES:
                logger.error(
                    "AWS SES sandbox/restriction error [%s] sending email to %s: %s",
                    error_code, to_email, error_message,
                )
            else:
                logger.error(
                    "AWS SES ClientError [%s] sending email to %s: %s",
                    error_code, to_email, error_message,
                )
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending email via SES to {to_email}: {str(e)}")
            return False

    def send_invite_email(self, email: str, invite_token: str, inviter_name: str, tenant_name: str) -> bool:
        """
        Send team invitation email

        Args:
            email: Invitee's email address
            invite_token: Invitation token
            inviter_name: Name of person sending invite
            tenant_name: Name of the team/tenant

        Returns:
            bool: True if email sent successfully, False otherwise
        """
        try:
            subject, html_body = build_invite_email_content(
                invite_token, inviter_name, tenant_name
            )
            return self._send_email(to_email=email, subject=subject, html_body=html_body)
        except Exception as e:
            logger.error(f"Error sending invite email to {email}: {str(e)}")
            return False

    def send_data_export_ready_email(
        self, email: str, download_url: str, workspace_name: str
    ) -> bool:
        """Notify the workspace admin that their GDPR data export is ready for download."""
        try:
            subject = f"Your data export for {workspace_name} is ready"
            body = f"""
            <html>
            <body>
                <h2>Your Data Export is Ready</h2>
                <p>Hello,</p>
                <p>The data export you requested for <strong>{workspace_name}</strong> has finished processing.</p>
                <p><a href="{download_url}" style="background-color: #28a745; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block;">Download Export</a></p>
                <p>Or copy and paste this link into your browser:</p>
                <p>{download_url}</p>
                <p><strong>This link will expire in 24 hours.</strong></p>
                <p>Best regards,<br>The Voice Agent Team</p>
            </body>
            </html>
            """
            return self._send_email(to_email=email, subject=subject, html_body=body)
        except Exception as e:
            logger.error(f"Error sending data export ready email to {email}: {str(e)}")
            return False

    def send_generic_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        cc_emails: Optional[List[str]] = None,
    ) -> bool:
        """
        Send a generic email, optionally CC'ing additional recipients.
        Used by features like sending call analyses or AI-generated summaries.
        """
        return self._send_email(
            to_email=to_email,
            subject=subject,
            html_body=html_body,
            cc_emails=cc_emails,
        )


# Create a global instance
email_service = EmailService()
