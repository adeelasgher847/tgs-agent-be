import logging
import certifi
import requests
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmailService:
    def __init__(self):
        self.api_key = settings.SENDGRID_API_KEY
        self.sender_email = settings.SENDGRID_SENDER_EMAIL

    def send_password_reset_email(self, email: str, reset_token: str, user_name: str) -> bool:
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = email
            msg['Subject'] = "Password Reset Request - Voice Agent Platform"

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
            msg.attach(MIMEText(body, 'html'))
            return self._send_email(msg)
        except Exception as e:
            logger.error(f"Error sending password reset email to {email}: {str(e)}")
            return False

    def send_invite_email(self, email: str, invite_token: str, inviter_name: str, tenant_name: str) -> bool:
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = email
            msg['Subject'] = f"You're invited to join {tenant_name} on Voice Agent Platform"

            invite_link = f"{settings.FRONTEND_URL}/accept-invite?token={invite_token}"
            body = f"""
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
            msg.attach(MIMEText(body, 'html'))
            return self._send_email(msg)
        except Exception as e:
            logger.error(f"Error sending invite email to {email}: {str(e)}")
            return False

    def _send_email(self, msg: MIMEMultipart) -> bool:
        if not self.api_key:
            logger.error("SendGrid API key is missing.")
            return False
        if not self.sender_email:
            logger.error("SENDGRID_SENDER_EMAIL is missing.")
            return False
        payload = self._build_payload(msg)
        try:
            response = requests.post(
                "https://api.sendgrid.com/v3/mail/send",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=10,
                verify=certifi.where(),
            )
            if 200 <= response.status_code < 300:
                logger.info(f"Email sent successfully to {msg['To']}")
                return True
            logger.error(
                "Failed to send email to %s. Status: %s, Body: %s",
                msg['To'],
                response.status_code,
                response.text,
            )
            return False
        except Exception as e:
            logger.error(f"Error sending email via SendGrid HTTP: {str(e)}")
            return False

    @staticmethod
    def _build_payload(msg: MIMEMultipart) -> dict:
        html_content = ""
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                html_content = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8")
                break
        return {
            "personalizations": [{"to": [{"email": msg['To']}]}],
            "from": {"email": msg['From']},
            "subject": msg['Subject'],
            "content": [{"type": "text/html", "value": html_content}],
        }


# Create a global instance
email_service = EmailService()
