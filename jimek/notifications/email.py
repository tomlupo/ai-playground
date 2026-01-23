"""Email notification adapter using SMTP."""

from __future__ import annotations

import logging
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import httpx

from jimek.notifications.base import NotificationAdapter, NotificationMessage

logger = logging.getLogger(__name__)


class EmailNotifier(NotificationAdapter):
    """
    Email notification adapter using SMTP.

    Supports:
        - SMTP with TLS/SSL
        - HTML and plain text emails
        - Multiple recipients
    """

    channel_name = "email"

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        username: Optional[str] = None,
        password: Optional[str] = None,
        from_addr: str = "jimek@localhost",
        to_addrs: Optional[list[str]] = None,
        use_tls: bool = True,
        use_ssl: bool = False,
        timeout: int = 30,
    ):
        """
        Initialize Email notifier.

        Args:
            smtp_host: SMTP server hostname
            smtp_port: SMTP server port
            username: SMTP authentication username
            password: SMTP authentication password
            from_addr: Sender email address
            to_addrs: List of recipient email addresses
            use_tls: Use STARTTLS
            use_ssl: Use SSL/TLS connection
            timeout: Connection timeout
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs or []
        self.use_tls = use_tls
        self.use_ssl = use_ssl
        self.timeout = timeout

    def _build_email(self, message: NotificationMessage) -> MIMEMultipart:
        """Build email message."""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"{message.emoji} {message.title}"
        msg["From"] = self.from_addr
        msg["To"] = ", ".join(self.to_addrs)

        # Plain text version
        text_content = message.format_plain()
        msg.attach(MIMEText(text_content, "plain"))

        # HTML version
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
                .content {{ margin: 15px 0; }}
                .footer {{ color: #666; font-size: 12px; margin-top: 20px; }}
                .level-info {{ color: #0088ff; }}
                .level-success {{ color: #00aa00; }}
                .level-warning {{ color: #ffaa00; }}
                .level-error {{ color: #ff0000; }}
            </style>
        </head>
        <body>
            <div class="header level-{message.level}">
                {message.emoji} {message.title}
            </div>
            <div class="content">
                {message.message.replace(chr(10), '<br>')}
            </div>
            <div class="footer">
                <p>Time: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                {f'<p>Job: {message.job_name}</p>' if message.job_name else ''}
                <p>Sent by Jimek Orchestrator</p>
            </div>
        </body>
        </html>
        """
        msg.attach(MIMEText(html_content, "html"))

        return msg

    def send(self, message: NotificationMessage) -> bool:
        """Send notification via email."""
        if not self.to_addrs:
            logger.warning("No email recipients configured")
            return False

        try:
            msg = self._build_email(message)

            if self.use_ssl:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(
                    self.smtp_host, self.smtp_port, context=context, timeout=self.timeout
                ) as server:
                    if self.username and self.password:
                        server.login(self.username, self.password)
                    server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
            else:
                with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=self.timeout) as server:
                    if self.use_tls:
                        context = ssl.create_default_context()
                        server.starttls(context=context)
                    if self.username and self.password:
                        server.login(self.username, self.password)
                    server.sendmail(self.from_addr, self.to_addrs, msg.as_string())

            logger.info(f"Email notification sent: {message.title}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False

    async def send_async(self, message: NotificationMessage) -> bool:
        """
        Send notification via email asynchronously.

        Note: Uses sync SMTP in thread pool as aiosmtplib adds complexity.
        For true async, consider using aiosmtplib.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.send(message))
