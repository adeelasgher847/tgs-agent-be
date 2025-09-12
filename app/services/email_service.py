import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        self.smtp_host = settings.SMTP_HOST
        self.smtp_port = settings.SMTP_PORT
        self.smtp_username = settings.SMTP_USERNAME
        self.smtp_password = settings.SMTP_PASSWORD
        self.smtp_tls = settings.SMTP_TLS
        self.smtp_ssl = settings.SMTP_SSL
    
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
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.smtp_username
            msg['To'] = email
            msg['Subject'] = "Password Reset Request - Voice Agent Platform"
            
            # Create reset link
            reset_link = f"{settings.FRONTEND_URL}/reset-password?token={reset_token}"
            
            # Email body
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
            
            # Send email
            return self._send_email(msg)
            
        except Exception as e:
            logger.error(f"Error sending password reset email to {email}: {str(e)}")
            return False
    
    def _send_email(self, msg: MIMEMultipart) -> bool:
        """
        Send email using SMTP
        
        Args:
            msg: MIME message to send
            
        Returns:
            bool: True if email sent successfully, False otherwise
        """
        try:
            if self.smtp_ssl:
                server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port)
            else:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            
            if self.smtp_tls:
                server.starttls()
            
            if self.smtp_username and self.smtp_password:
                server.login(self.smtp_username, self.smtp_password)
            
            text = msg.as_string()
            server.sendmail(msg['From'], msg['To'], text)
            server.quit()
            
            logger.info(f"Email sent successfully to {msg['To']}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
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
            msg = MIMEMultipart()
            msg['From'] = self.smtp_username
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

# Create a global instance
email_service = EmailService()
