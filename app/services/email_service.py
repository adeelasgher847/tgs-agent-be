import logging
from typing import List, Optional

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from app.core.config import settings

logger = logging.getLogger(__name__)


class EmailService:
    def __init__(self):
        self.sg_client = SendGridAPIClient(settings.SENDGRID_API_KEY) if settings.SENDGRID_API_KEY else None
        self.sender_email = settings.SENDGRID_SENDER_EMAIL
    
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
    ) -> bool:
        """
        Send an email using SendGrid API.
        """
        try:
            if not self.sg_client:
                logger.error("SendGrid client is not configured. Missing SENDGRID_API_KEY.")
                return False
            message = Mail(
                from_email=self.sender_email,
                to_emails=to_email,
                subject=subject,
                html_content=html_body,
            )
            # Optionally add CC recipients
            if cc_emails:
                for cc in cc_emails:
                    if cc:
                        message.add_cc(cc)
            response = self.sg_client.send(message)
            if 200 <= response.status_code < 300:
                logger.info(f"Email sent successfully to {to_email}")
                return True
            logger.error(f"Failed to send email to {to_email}. Status: {response.status_code}")
            return False
        except Exception as e:
            logger.error(f"Error sending email via SendGrid: {str(e)}")
            return False
    
    def send_invite_email(self, email: str, invite_token: str, inviter_name: str, tenant_name: str) -> bool:
        """
        Send team invitation email via Gmail
        
        Args:
            email: Invitee's email address
            invite_token: Invitation token
            inviter_name: Name of person sending invite
            tenant_name: Name of the team/tenant
            
        Returns:
            bool: True if email sent successfully, False otherwise
        """
        try:
            subject = f"You're invited to join {tenant_name} on Voice Agent Platform"
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
            return self._send_email(to_email=email, subject=subject, html_body=body)
        except Exception as e:
            logger.error(f"Error sending invite email to {email}: {str(e)}")
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
